"""Train and score two separate models: a classifier (is a molecule active, pKi >= 7?) and a
regressor (what is its pKi?).

A researcher (or the team) chooses the features, any scikit-learn model on the menu and the
settings to try. Everything that decides how honest
the scores are is fixed here: the splits and folds (made by cb2.split), the seed, how features
are prepared, the metrics, the baselines and the test part. Choices are compared on practice
scores from cross-validation inside the training part. Only `evaluate` reads the test part, and
it logs every time it does, so the final scores come from molecules nothing was tuned on."""

import itertools
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
from scipy.stats import rankdata, spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

from cb2.checks import CheckFailed, sha256, verify_files
from cb2.clean import ROOT, code_version, shown
from cb2.features import FEATURES, featurize
from cb2.split import ACTIVE_AT, FOLDS, GROUPS, SEED

MODELS_DIR = ROOT / "data" / "models"
TEST_LOG = ROOT / "data" / "test-use.jsonl"
HEADLINE = "scaffold"
# The classifier is also checked at these cutoffs, with the settings chosen at ACTIVE_AT
SENSITIVITY = (6.0, 8.0)
BOOTSTRAPS = 1000
# The score that picks each model's settings, and whether higher is better
PICK_BY = {"classifier": ("roc_auc", True), "regressor": ("rmse", False)}
# Set by the project wherever a model has them, never chosen
FIXED = {"random_state": SEED}
# The most combinations of settings one choice may try, so a run stays a sensible length
MAX_SETTINGS = 12
# How long one of the team's tries may take, in minutes
TRY_MINUTES = 10


def build(estimator: type, setting: dict):
    """Every model sees the features the same way: missing values filled with the training
    median, then each feature put on the same scale. The seed is set wherever a model takes one."""
    fixed = {key: value for key, value in FIXED.items() if key in estimator().get_params()}
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         estimator(**fixed, **setting))


@lru_cache(maxsize=2)
def menu(task: str) -> dict[str, type]:
    """Every scikit-learn model of this kind that can be built with its defaults and passes a
    quick fit-and-predict check through `build`. Dummy models are the baselines, so they're left out."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(40, 6))
    y = (X[:, 0] > 0).astype(int) if task == "classifier" else X[:, 0] + 0.1 * rng.normal(size=40)
    found = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, estimator in all_estimators(type_filter=task):
            try:
                if not name.startswith("Dummy") and np.isfinite(
                        predict(build(estimator, {}).fit(X[:30], y[:30]), X[30:], task)).all():
                    found[name] = estimator
            except Exception:
                continue
    return found


@dataclass(frozen=True)
class Choice:
    """One model choice: a scikit-learn model and the settings to try (name -> list of values)."""
    task: str
    model: str
    settings: dict


class ChoiceError(ValueError):
    pass


class TooSlow(ChoiceError):
    pass


def duration(seconds: float) -> str:
    if seconds < 1:
        return "under a second"
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60:.0f} minutes" if seconds < 5400 else f"{seconds / 3600:.1f} hours"


class Clock:
    """Times a try's fits. After each one it projects the total from the average so far, and stops
    the try as soon as that would run past the limit, rather than finding out at the end."""

    def __init__(self, fits: int, minutes: float | None = None):
        self.fits, self.minutes, self.start, self.done = fits, minutes, time.monotonic(), 0

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.start

    def tick(self) -> None:
        self.done += 1
        each = self.seconds / self.done
        if self.minutes is not None and each * self.fits > 60 * self.minutes:
            raise TooSlow(f"this would take about {duration(each * self.fits)} ({self.fits} fits at about "
                          f"{duration(each)} each), over the {self.minutes:g}-minute limit for "
                          f"one try; try fewer settings, settings that train faster, or a faster model")


def check_features(features: str) -> None:
    if features not in FEATURES:
        raise ChoiceError(f"features must be one of {list(FEATURES)}")


def make_choice(task: str, model: str, settings: dict | None = None) -> Choice:
    """A choice of model and settings to try, refusing anything off the menu. A setting given as
    a single value means "just this one"; settings not given keep the model's defaults."""
    if task not in PICK_BY:
        raise ChoiceError(f"task must be one of {list(PICK_BY)}")
    grid, available = {}, menu(task)
    if model not in available:
        raise ChoiceError(f"{task} must be one of the {len(available)} scikit-learn {task}s on the menu "
                          f"(python -m cb2.train --menu), not {model!r}")
    if not isinstance(settings or {}, dict):
        raise ChoiceError("settings must be an object of setting name -> value or list of values")
    known, problems = available[model]().get_params(), []
    for name, values in (settings or {}).items():
        values = values if isinstance(values, list) else [values]
        if name in FIXED:
            problems.append(f"{name} is set by the project")
        elif name not in known:
            problems.append(f"{model} has no setting {name!r}")
        elif not values or not all(v is None or isinstance(v, (bool, int, float, str)) for v in values):
            problems.append(f"{name} must be a number, text, true/false or null, or a list of them")
        grid = {**grid, name: values}
    if (combinations := math.prod(len(values) for values in grid.values())) > MAX_SETTINGS:
        problems.append(f"that is {combinations} combinations of settings; the most is {MAX_SETTINGS}")
    if problems:
        raise ChoiceError("; ".join(problems))
    return Choice(task, model, grid)


class NearestNeighbour:
    """Copies the value of the most similar training molecule (Tanimoto similarity of Morgan
    fingerprint bits). A model is only interesting if it beats this."""

    def fit(self, X, y):
        self.X, self.y = X, np.asarray(y)
        return self

    def predict(self, X):
        shared = X @ self.X.T
        either = X.sum(1)[:, None] + self.X.sum(1)[None, :] - shared
        return self.y[np.argmax(shared / np.maximum(either, 1), axis=1)]

    def predict_proba(self, X):
        active = self.predict(X).astype(float)
        return np.column_stack([1 - active, active])


# name: (description, {task: make a model}, the features it uses)
BASELINES = {
    "majority / mean": ("always predicts the commonest class, or the average pKi",
                        {"classifier": lambda: DummyClassifier(strategy="prior"),
                         "regressor": lambda: DummyRegressor(strategy="mean")}, "morgan2"),
    "nearest neighbour": ("copies the most similar training molecule",
                          {"classifier": NearestNeighbour, "regressor": NearestNeighbour}, "bits"),
}


@lru_cache(maxsize=8)
def matrix(smiles: tuple[str, ...], features: str) -> np.ndarray:
    if features == "bits":
        return (featurize(list(smiles), "morgan2") > 0).astype(np.float32)
    return featurize(list(smiles), features)


def load_split(folder: Path) -> tuple[pd.DataFrame, dict]:
    manifest = verify_files(folder, f"The split in {folder}")
    table = pd.read_csv(folder / "molecules.csv")
    if len(table) != manifest["rows"]["molecules"]:
        raise CheckFailed(f"The split in {folder}", [f"molecules.csv has {len(table):,} rows but the manifest says "
                                                     f"{manifest['rows']['molecules']:,}"])
    return table, manifest


def training_part(table: pd.DataFrame, split: str) -> pd.DataFrame:
    return table[table[f"{split}_part"] == "train"].reset_index(drop=True)


def labels(rows: pd.DataFrame, task: str, cutoff: float = ACTIVE_AT) -> np.ndarray:
    return (rows["pchembl"] >= cutoff).to_numpy(int) if task == "classifier" else rows["pchembl"].to_numpy(float)


def predict(model, X, task: str) -> np.ndarray:
    """A regressor's values; for a classifier, two columns: a score to rank by (the probability
    of active, or the model's own score if it has no probabilities) and its yes/no call."""
    # A model running on several threads (a forest) adds up its parts in whatever order the threads
    # finish, which changes the last digits from run to run; predicting on one thread keeps them identical
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    if "n_jobs" in getattr(last, "get_params", dict)():
        last.set_params(n_jobs=1)
    if task == "regressor":
        return model.predict(X)
    called = model.predict(X).astype(float)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X) if hasattr(model, "decision_function") else called
    return np.column_stack([score, called])


def classifier_scores(y: np.ndarray, predicted: np.ndarray) -> dict:
    # Written out rather than with sklearn's functions, which are too slow for 1,000 bootstrap
    # samples; the tests check they agree
    predicted, called = predicted[:, 0], predicted[:, 1] == 1
    positive = y == 1
    n1, n0 = positive.sum(), (~positive).sum()
    tp, fn = (called & positive).sum(), (~called & positive).sum()
    fp, tn = (called & ~positive).sum(), (~called & ~positive).sum()
    both = n1 > 0 and n0 > 0
    # ROC AUC: the chance a random active is scored above a random inactive (ties count half)
    auc = (rankdata(predicted)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if both else np.nan
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"roc_auc": float(auc), "average_precision": average_precision_score(y, predicted) if both else np.nan,
            "balanced_accuracy": float(np.nanmean([tp / n1 if n1 else np.nan, tn / n0 if n0 else np.nan])),
            "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0}


def scores(task: str, y: np.ndarray, predicted: np.ndarray) -> dict:
    with warnings.catch_warnings():
        # A constant prediction (the baselines) or a one-class sample gives NaN scores, not errors
        warnings.simplefilter("ignore")
        if task == "classifier":
            return classifier_scores(y, predicted)
        return {"rmse": root_mean_squared_error(y, predicted), "mae": mean_absolute_error(y, predicted),
                "r2": r2_score(y, predicted), "spearman": spearmanr(y, predicted).statistic}


def settings(grid: dict) -> list[dict]:
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def cross_validate(train: pd.DataFrame, split: str, task: str, make, features: str, cutoff: float = ACTIVE_AT,
                   clock: Clock | None = None) -> dict:
    """Mean and spread of each score over the fixed folds; each fold is scored by a model that
    never saw it."""
    X, y = matrix(tuple(train["smiles"]), features), labels(train, task, cutoff)
    folds = train[f"{split}_fold"].to_numpy(int)
    per_fold = []
    for k in range(FOLDS):
        model = make().fit(X[folds != k], y[folds != k])
        if clock:
            clock.tick()
        per_fold.append(scores(task, y[folds == k], predict(model, X[folds == k], task)))
    return {name: {"mean": float(np.mean(values)), "sd": float(np.std(values))}
            for name in per_fold[0] for values in [[fold[name] for fold in per_fold]]}


def practice(train: pd.DataFrame, split: str, features: str, choice: Choice, cutoff: float = ACTIVE_AT,
             minutes: float | None = None) -> dict:
    """Tries each combination of the choice's settings and keeps the best by practice score.
    With a limit in minutes, stops as soon as it's clear the whole try would take longer."""
    estimator, (metric, higher) = menu(choice.task)[choice.model], PICK_BY[choice.task]
    combinations, tried = settings(choice.settings), []
    clock = Clock(len(combinations) * FOLDS, minutes)
    for setting in combinations:
        try:
            tried.append({"settings": setting, "scores": cross_validate(
                train, split, choice.task, lambda: build(estimator, setting), features, cutoff, clock)})
        except TooSlow as error:
            raise TooSlow(f"{choice.model}: {error}") from None
        except Exception as error:
            raise ChoiceError(f"{choice.model} with {setting} failed: {error}") from error
    # A score that can't be computed (NaN) counts as the worst
    best = max(tried, key=lambda t: np.nan_to_num((1 if higher else -1) * t["scores"][metric]["mean"], nan=-np.inf))
    return {"model": choice.model, "features": features, "settings": best["settings"], "scores": best["scores"],
            "tried": len(tried), "each": [{"settings": t["settings"], metric: t["scores"][metric]["mean"]} for t in tried],
            "seconds": round(clock.seconds, 1)}


def practice_baselines(train: pd.DataFrame, split: str, task: str, cutoff: float = ACTIVE_AT) -> dict:
    return {name: cross_validate(train, split, task, make[task], features, cutoff)
            for name, (_, make, features) in BASELINES.items()}


def count_test_uses(folder: Path) -> int:
    if not TEST_LOG.exists():
        return 0
    return sum(json.loads(line)["split_folder"] == shown(folder) for line in TEST_LOG.read_text().splitlines())


def open_test_part(table: pd.DataFrame, split: str, folder: Path, purpose: dict) -> pd.DataFrame:
    """The only way to the test part: every use is logged before the rows are handed over."""
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split_folder": shown(folder),
             "split": split, **purpose}
    with TEST_LOG.open("a") as log:
        log.write(json.dumps(entry) + "\n")
    return table[table[f"{split}_part"] == "test"].reset_index(drop=True)


def with_interval(task: str, y: np.ndarray, predicted: np.ndarray) -> dict:
    """Each test score with a 95% bootstrap interval: the range it would likely cover on another
    test set of the same size."""
    rng = np.random.default_rng(SEED)
    draws = [scores(task, y[i], predicted[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAPS))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a score that is NaN in every sample has a NaN interval
        return {name: {"value": float(value), "low": float(np.nanpercentile([d[name] for d in draws], 2.5)),
                       "high": float(np.nanpercentile([d[name] for d in draws], 97.5))}
                for name, value in scores(task, y, predicted).items()}


def targets(classifier: Choice, regressor: Choice) -> list[tuple[str, str, Choice, float | None]]:
    # (label, task, choice, cutoff); the classifier at other cutoffs reuses its settings from 7
    return ([(f"active at {ACTIVE_AT:g}", "classifier", classifier, ACTIVE_AT)]
            + [(f"active at {cutoff:g}", "classifier", classifier, cutoff) for cutoff in SENSITIVITY]
            + [("pKi", "regressor", regressor, None)])


def evaluate(table: pd.DataFrame, folder: Path, features: str, classifier: Choice, regressor: Choice):
    """Pick settings on practice scores, fit on the whole training part, then open the test
    part once per split and score the models and baselines on it."""
    check_features(features)
    results, predictions, fitted = {}, [], {}
    for split in GROUPS:
        train = training_part(table, split)
        chosen, models = {}, {}
        for label, task, choice, cutoff in targets(classifier, regressor):
            if task == "regressor" or cutoff == ACTIVE_AT:
                chosen[task] = practice(train, split, features, choice)
            models[label] = build(menu(task)[choice.model], chosen[task]["settings"]).fit(
                matrix(tuple(train["smiles"]), features), labels(train, task, cutoff or ACTIVE_AT))
        test = open_test_part(table, split, folder, {"features": features, "classifier": classifier.model,
                                                     "regressor": regressor.model})
        results[split] = {}
        for label, task, choice, cutoff in targets(classifier, regressor):
            y = labels(test, task, cutoff or ACTIVE_AT)
            predicted = predict(models[label], matrix(tuple(test["smiles"]), features), task)
            baselines = {}
            for name, (_, make, base_features) in BASELINES.items():
                base = make[task]().fit(matrix(tuple(train["smiles"]), base_features), labels(train, task, cutoff or ACTIVE_AT))
                baselines[name] = with_interval(task, y, predict(base, matrix(tuple(test["smiles"]), base_features), task))
            results[split][label] = {"model": choice.model, "settings": chosen[task]["settings"],
                                     "practice": chosen[task]["scores"] if cutoff in (None, ACTIVE_AT) else None,
                                     "test": with_interval(task, y, predicted), "baselines": baselines}
            predictions.append(pd.DataFrame({"compound_id": test["compound_id"], "split": split, "target": label,
                                             "observed": y, "predicted": predicted if task == "regressor" else
                                             predicted[:, 0], "called_active": None if task == "regressor" else
                                             predicted[:, 1]}))
        fitted[split] = models
    return results, pd.concat(predictions, ignore_index=True), fitted


def practice_text(results: dict, baselines: dict) -> str:
    metrics = list(next(iter(results.values()))["scores"])
    lines = [f"{'':<32}" + "".join(f"{name:>20}" for name in metrics)]
    for name, result in list(results.items()) + list(baselines.items()):
        row = result["scores"] if "scores" in result else result
        lines.append(f"{name:<32}" + "".join(f"{row[m]['mean']:>13.3f} ± {row[m]['sd']:.2f}" for m in metrics))
    return "\n".join(lines)


def results_text(results: dict) -> str:
    lines = []
    for split, targets_ in results.items():
        lines.append(f"{split} split (test part)")
        for label, result in targets_.items():
            for name, values in [(result["model"], result["test"])] + list(result["baselines"].items()):
                lines.append(f"  {label:<14}{name:<32}" + "  ".join(
                    f"{metric} {v['value']:.3f} [{v['low']:.3f}, {v['high']:.3f}]" for metric, v in values.items()))
    return "\n".join(lines)


def tidy(value):
    # NaN (a score that can't be computed, e.g. for a constant prediction) is written as null
    if isinstance(value, dict):
        return {key: tidy(item) for key, item in value.items()}
    return None if isinstance(value, float) and np.isnan(value) else value


def save(results: dict, predictions: pd.DataFrame, fitted: dict, source: Path, features: str, classifier: Choice,
         regressor: Choice, uses_before: int, run_id: str = "", reason: str = "") -> Path:
    folder = MODELS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder.mkdir(parents=True)
    (folder / "scores.json").write_text(json.dumps(tidy(results), indent=2))
    predictions.to_csv(folder / "predictions.csv", index=False)
    for label, model in fitted[HEADLINE].items():
        joblib.dump(model, folder / f"{HEADLINE} {label}.joblib")
    manifest = {
        "made_at_utc": folder.name,
        "run": run_id,
        "source": {"folder": shown(source), "manifest_sha256": sha256(source / "manifest.json")},
        "choice": {"features": features, "classifier": asdict(classifier), "regressor": asdict(regressor)},
        "reason": reason,
        "settings": {"seed": SEED, "active_at": ACTIVE_AT, "sensitivity": list(SENSITIVITY), "folds": FOLDS,
                     "bootstraps": BOOTSTRAPS, "pick_by": PICK_BY, "fixed": FIXED, "max_settings": MAX_SETTINGS,
                     "features_prepared": "missing values filled with the training median, then standard scaling"},
        "versions": {"scikit-learn": sklearn.__version__, "rdkit": rdkit.__version__, "code": code_version()},
        "test_uses_before": uses_before,
        "sha256": {path.name: sha256(path) for path in sorted(folder.iterdir())},
    }
    # Written last, so a folder without a manifest is unfinished
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder


# --- Tools for the team: practice scores on the training part, and a proposal ---

CHOICE_PROPERTIES = {
    "model": {"type": "string", "description": "A scikit-learn model name from the menu"},
    "settings": {"type": "object", "description": ('Settings to try, name -> a value or a list of values, e.g. '
                                                   '{"C": [0.1, 1, 10]}. Settings left out keep the defaults')},
}

TRY_TOOL = {
    "name": "try_models",
    "description": (f"Practice scores for one model: {FOLDS}-fold cross-validation on the training part of the "
                    f"{HEADLINE} split, trying each combination of settings and keeping the best, next to the "
                    "baselines. The test part is never used."),
    "input_schema": {"type": "object", "properties": {"features": {"type": "string", "enum": list(FEATURES)},
                                                      "task": {"type": "string", "enum": list(PICK_BY)},
                                                      **CHOICE_PROPERTIES},
                     "required": ["features", "task", "model"]},
}

PROPOSE_TOOL = {
    "name": "propose_models",
    "description": ("Propose the features, the classifier and the regressor, each with the settings to try. Once the "
                    "project lead approves, code picks the settings on practice scores, trains on the whole training "
                    "part and scores the test part once. Calling it again replaces the earlier proposal."),
    "input_schema": {"type": "object", "properties": {
        "features": {"type": "string", "enum": list(FEATURES)},
        **{task: {"type": "object", "properties": CHOICE_PROPERTIES, "required": ["model"]} for task in PICK_BY},
        "reason": {"type": "string", "description": "Why these choices"}},
        "required": ["features", *PICK_BY, "reason"]},
}


@lru_cache(maxsize=64)
def cached_practice(folder: str, features: str, task: str, model: str, grid: str, minutes: float | None) -> dict:
    # The same try made twice in a run (by the planner and the reviewer, say) is only computed once
    train = training_part(load_split(Path(folder))[0], HEADLINE)
    return practice(train, HEADLINE, features, Choice(task, model, json.loads(grid)), minutes=minutes)


@lru_cache(maxsize=4)
def cached_baselines(folder: str, task: str) -> dict:
    return practice_baselines(training_part(load_split(Path(folder))[0], HEADLINE), HEADLINE, task)


def practice_for(folder: Path, features: str, choice: Choice, minutes: float | None) -> tuple[dict, dict]:
    check_features(features)
    return (cached_practice(str(folder), features, choice.task, choice.model, json.dumps(choice.settings), minutes),
            cached_baselines(str(folder), choice.task))


def bootstrap_seconds(tested: int) -> float:
    """How long the 95% intervals take, timed here: every split, target, model and baseline is
    scored BOOTSTRAPS times on a test part of this size."""
    rng, each = np.random.default_rng(SEED), {}
    y = rng.normal(7, 1, tested)
    for task, predicted in (("classifier", np.column_stack([rng.random(tested), rng.random(tested) > 0.5])),
                            ("regressor", y + rng.normal(0, 1, tested))):
        start = time.monotonic()
        for _ in range(20):
            scores(task, (y >= 7).astype(int) if task == "classifier" else y, predicted)
        each[task] = (time.monotonic() - start) / 20
    per_target = (1 + len(SENSITIVITY)) * each["classifier"] + each["regressor"]
    return len(GROUPS) * (1 + len(BASELINES)) * BOOTSTRAPS * per_target


def evaluate_seconds(classifier: dict, regressor: dict, tested: int) -> float:
    """A rough time for `evaluate` from the practice timings: for each split, practice again, then
    one final fit per target (three classifiers, one regressor), each about the time of one
    practice fit; then the intervals."""
    per_fit = {task: result["seconds"] / (result["tried"] * FOLDS) for task, result in
               (("classifier", classifier), ("regressor", regressor))}
    return (len(GROUPS) * (classifier["seconds"] + regressor["seconds"]
                           + (1 + len(SENSITIVITY)) * per_fit["classifier"] + per_fit["regressor"])
            + bootstrap_seconds(tested))


def numbered(items: dict) -> dict:
    # Ledger numbers for the auditor; a score that can't be computed has nothing to check
    return {name: value for name, value in items.items() if value is not None and np.isfinite(value)}


def try_models(folder: Path, request: dict, look_id: str, minutes: float | None = TRY_MINUTES) -> dict:
    """Practice scores as a ledger entry, so the auditor can check numbers quoted from them."""
    features = str(request.get("features"))
    choice = make_choice(str(request.get("task")), str(request.get("model")), request.get("settings"))
    result, baselines = practice_for(folder, features, choice, minutes)
    metric = PICK_BY[choice.task][0]
    rows = {choice.model: result["scores"], **baselines}
    numbers = numbered({f"{name} | {score} {stat}": value for name, row in rows.items()
                        for score, values in row.items() for stat, value in values.items()}
                       | {f"setting {i} | {metric}": tried[metric] for i, tried in enumerate(result["each"], 1)})
    each = "\n".join(f"  {i}. {tried['settings'] or 'defaults'}: {metric} {tried[metric]:.3f}"
                     for i, tried in enumerate(result["each"], 1))
    text = (f"{look_id} try_models {choice.task} {choice.model} on {features}, {HEADLINE} split, training part "
            f"(mean ± sd over {FOLDS} folds), best settings {result['settings'] or 'defaults'}\n"
            f"{practice_text({choice.model: result}, baselines)}\n"
            f"Each of the {result['tried']} settings tried:\n{each}\n"
            f"This try took {duration(result['seconds'])}.")
    return {"id": look_id, "request": {"kind": "try_models", **request}, "text": text, "numbers": numbers}


def propose(folder: Path, request: dict, minutes: float | None = TRY_MINUTES) -> dict:
    """A proposal checked against the menu, with practice scores for the lead and how long training
    and testing should take, computed by code."""
    features, proposal, tables, timings = str(request.get("features")), {}, [], {}
    for task in PICK_BY:
        part = request.get(task) if isinstance(request.get(task), dict) else {"model": request.get(task)}
        choice = make_choice(task, str(part.get("model")), part.get("settings"))
        result, baselines = practice_for(folder, features, choice, minutes)
        proposal[task], timings[task] = asdict(choice), result
        tables.append(f"{task}, best of {result['tried']} settings: {result['settings'] or 'defaults'}\n"
                      + practice_text({choice.model: result}, baselines))
    heading = f"Practice scores ({HEADLINE} split, training part, mean ± sd over {FOLDS} folds):\n\n"
    tested = int((load_split(folder)[0][f"{HEADLINE}_part"] == "test").sum())
    seconds = evaluate_seconds(**timings, tested=tested)
    estimate = (f"\n\nTraining and testing should take about {duration(seconds)} on this computer (a rough "
                "estimate from the practice timings). It runs here and has no API cost.")
    return {"features": features, **proposal, "reason": str(request.get("reason", "")),
            "practice": heading + "\n\n".join(tables) + estimate}


def choices(proposal: dict) -> tuple[Choice, Choice]:
    return tuple(make_choice(task, proposal[task]["model"], proposal[task]["settings"]) for task in PICK_BY)


def scores_entry(results: dict, look_id: str) -> dict:
    """The test scores as a ledger entry, so the conclusions can be checked against them."""
    numbers = {}
    for split, targets_ in results.items():
        for label, result in targets_.items():
            rows = {result["model"]: result["test"], **result["baselines"]}
            numbers |= {f"{split} | {label} | {name} | {metric}{'' if bound == 'value' else ' ' + bound}": value
                        for name, row in rows.items() for metric, values in row.items()
                        for bound, value in values.items()}
            numbers |= {f"{split} | {label} | {result['model']} | practice {metric} {stat}": value
                        for metric, values in (result["practice"] or {}).items() for stat, value in values.items()}
    return {"id": look_id, "request": {"kind": "test_scores"}, "numbers": numbered(numbers),
            "text": f"{look_id} test scores, each with its 95% interval\n{results_text(results)}"}


def menu_text() -> str:
    return "\n\n".join(f"{task}s ({len(menu(task))}):\n  " + ", ".join(menu(task))
                         for task in ("classifier", "regressor"))


if __name__ == "__main__":
    # Usage: python -m cb2.train --menu
    #        python -m cb2.train data/split/<folder> --features morgan2 --classifier RandomForestClassifier
    #            --regressor SVR --regressor-settings '{"C": [1, 10]}' [--practice]
    import argparse

    parser = argparse.ArgumentParser(description="Train and score the classifier and the regressor.")
    parser.add_argument("split", type=Path, nargs="?")
    parser.add_argument("--menu", action="store_true", help="list every model on the menu")
    parser.add_argument("--features", choices=list(FEATURES))
    for task in ("classifier", "regressor"):
        parser.add_argument(f"--{task}", help="a scikit-learn model name from the menu")
        parser.add_argument(f"--{task}-settings", type=json.loads, default=None,
                            help='settings to try as JSON, e.g. \'{"C": [0.1, 1, 10]}\'')
    parser.add_argument("--practice", action="store_true",
                        help="only show practice scores on the training part; the test part stays closed")
    args = parser.parse_args()
    if args.menu:
        print(menu_text())
        raise SystemExit
    if not (args.split and args.features and args.classifier and args.regressor):
        parser.error("give a split folder, --features, --classifier and --regressor (or --menu)")
    try:
        check_features(args.features)
        classifier = make_choice("classifier", args.classifier, args.classifier_settings)
        regressor = make_choice("regressor", args.regressor, args.regressor_settings)
    except ChoiceError as error:
        parser.error(str(error))
    table, _ = load_split(args.split)
    train = training_part(table, HEADLINE)
    if args.practice:
        for choice in (classifier, regressor):
            result = practice(train, HEADLINE, args.features, choice)
            print(f"{choice.task}, {HEADLINE} split, practice (mean ± sd over {FOLDS} folds), "
                  f"best of {result['tried']} settings: {result['settings']}, took {duration(result['seconds'])}")
            print(practice_text({choice.model: result}, practice_baselines(train, HEADLINE, choice.task)) + "\n")
    else:
        uses = count_test_uses(args.split.resolve())
        if uses:
            print(f"Note: this split's test part has been opened {uses} time(s) before.")
        results, predictions, fitted = evaluate(table, args.split.resolve(), args.features, classifier, regressor)
        print(results_text(results))
        print(f"Saved to {save(results, predictions, fitted, args.split.resolve(), args.features, classifier, regressor, uses)}")

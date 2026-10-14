import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from cb2 import split, train
from cb2.checks import CheckFailed


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    # Fewer bootstrap samples keep the tests fast; the code paths are the same
    monkeypatch.setattr(train, "BOOTSTRAPS", 100)


@pytest.fixture
def table(clean):
    return split.make_splits(split.prepare(clean)[0])


@pytest.fixture
def saved(table, tmp_path, monkeypatch):
    """A saved split, with the test-use log and model folder kept in tmp_path."""
    source = tmp_path / "clean"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setattr(split, "SPLIT", tmp_path / "split")
    monkeypatch.setattr(train, "TEST_LOG", tmp_path / "test-use.jsonl")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    return split.save(table, pd.DataFrame(columns=["compound", "reason"]), source)


LINEAR = {"classifier": ("LogisticRegression", {"C": [0.1, 1.0]}), "regressor": ("Ridge", {"alpha": [1.0, 10.0]})}


def choice(task):
    return train.make_choice(task, *LINEAR[task])


def untimed(result: dict) -> str:
    # Compared as JSON so that NaN (a score a tiny fold can't give) equals NaN; how long it took isn't a result
    return json.dumps({key: value for key, value in result.items() if key != "seconds"})


def test_the_menu_lists_scikit_learn_models_except_the_baselines():
    classifiers, regressors = train.menu("classifier"), train.menu("regressor")
    assert {"RandomForestClassifier", "LogisticRegression", "SVC", "KNeighborsClassifier", "MLPClassifier"} <= set(classifiers)
    assert {"RandomForestRegressor", "Ridge", "SVR", "KNeighborsRegressor", "MLPRegressor"} <= set(regressors)
    assert "DummyClassifier" not in classifiers and "DummyRegressor" not in regressors
    assert {"XGBClassifier", "TanimotoSVC", "TorchNetClassifier"} <= set(classifiers)
    assert {"XGBRegressor", "TanimotoSVR", "TorchNetRegressor"} <= set(regressors)
    # The project's own models are listed with their settings, so they can be chosen without reading the code
    assert ("TorchNetRegressor: batch_size=64, dropout=0.2, epochs=30, hidden_layers='256', learning_rate=0.001"
            in train.menu_text())


def test_choices_are_checked_against_the_menu():
    made = train.make_choice("classifier", "SVC", {"C": [0.1, 1, 10], "kernel": "rbf"})
    assert made.settings == {"C": [0.1, 1, 10], "kernel": ["rbf"]} and len(train.settings(made.settings)) == 3
    for task, model, settings, message in [
        ("regressor", "DeepMagicRegressor", None, "must be one of the"),
        ("classifier", "Ridge", None, "must be one of the"),
        ("classifier", "SVC", {"gamma_ray": [1]}, "has no setting 'gamma_ray'"),
        ("classifier", "RandomForestClassifier", {"random_state": [1]}, "random_state is set by the project"),
        ("classifier", "SVC", {"C": [{"nested": 1}]}, "must be a number"),
        ("classifier", "SVC", {"C": [1, 2, 3, 4], "gamma": [1, 2, 3, 4]}, "16 combinations"),
    ]:
        with pytest.raises(train.ChoiceError, match=message):
            train.make_choice(task, model, settings)
    with pytest.raises(train.ChoiceError, match="features must be one of"):
        train.check_features("morgan9")


def test_nearest_neighbour_copies_the_most_similar_molecule():
    smiles = ("Cc1ccccc1", "CCCCO")
    model = train.NearestNeighbour().fit(train.matrix(smiles, "bits"), np.array([8.0, 5.0]))
    assert model.predict(train.matrix(("CCc1ccccc1", "CCCCCO"), "bits")).tolist() == [8.0, 5.0]


def test_tanimoto_compares_which_fingerprint_bits_are_set():
    X = np.array([[1, 1, 0, 0], [2, 1, 0, 0], [0, 0, 0, 0]])
    Y = np.array([[1, 0, 1, 0], [1, 1, 0, 0]])
    # Shared bits over bits set in either; counts only matter as present or not
    np.testing.assert_allclose(train.tanimoto(X, Y), [[1 / 3, 1], [1 / 3, 1], [0, 0]])
    # The SVMs compare the fingerprints themselves, so they skip the scaling step
    assert len(train.build(train.TanimotoSVC, {}).steps) == 2 and len(train.build(train.SVC, {}).steps) == 3


def test_tanimoto_models_refuse_features_that_are_not_fingerprints(table):
    made = train.make_choice("regressor", "TanimotoSVR")
    with pytest.raises(train.ChoiceError, match="needs morgan2 or morgan3 features"):
        train.practice(train.training_part(table, "scaffold"), "scaffold", "descriptors", made)


def test_scores_are_right_and_agree_with_sklearn():
    y, score = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    assert train.scores("classifier", y, np.column_stack([score, score >= 0.5])) == {
        "roc_auc": 1.0, "average_precision": 1.0, "balanced_accuracy": 1.0, "mcc": 1.0}
    rng = np.random.default_rng(0)
    for _ in range(20):
        y, predicted = rng.integers(0, 2, 50), rng.choice([0.2, 0.5, 0.7, 0.9], 50)
        assert train.scores("classifier", y, np.column_stack([predicted, predicted >= 0.5])) == pytest.approx({
            "roc_auc": roc_auc_score(y, predicted), "average_precision": average_precision_score(y, predicted),
            "balanced_accuracy": balanced_accuracy_score(y, predicted >= 0.5),
            "mcc": matthews_corrcoef(y, predicted >= 0.5)})
    regression = train.scores("regressor", np.array([5.0, 6.0, 7.0]), np.array([5.0, 6.0, 8.0]))
    assert regression["mae"] == pytest.approx(1 / 3) and regression["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert regression["spearman"] == pytest.approx(1.0)
    # A constant prediction can't be ranked: NaN, not an error
    assert np.isnan(train.scores("regressor", np.array([5.0, 6.0]), np.array([5.5, 5.5]))["spearman"])


@pytest.mark.parametrize("task, model, settings", [
    ("classifier", "RandomForestClassifier", {"n_estimators": 20, "min_samples_leaf": [1, 3]}),
    ("regressor", "RandomForestRegressor", {"n_estimators": 20, "min_samples_leaf": [1, 3]}),
    ("classifier", "HistGradientBoostingClassifier", {"max_iter": 20}),
    ("regressor", "KNeighborsRegressor", {"n_neighbors": [3, 5]}),
    ("classifier", "SVC", {"C": [0.1, 1.0]}),  # no probabilities: ranked by its own score
    ("classifier", "MLPClassifier", {"hidden_layer_sizes": [16], "max_iter": 50}),
    ("regressor", "MLPRegressor", {"hidden_layer_sizes": [16], "max_iter": 50}),
    ("classifier", "XGBClassifier", {"n_estimators": 20, "max_depth": [2, 4]}),
    ("regressor", "XGBRegressor", {"n_estimators": 20, "max_depth": [2, 4]}),
    ("classifier", "TanimotoSVC", {"C": [0.1, 1.0]}),
    ("regressor", "TanimotoSVR", {"C": [1.0, 10.0]}),
    ("classifier", "TorchNetClassifier", {"hidden_layers": ["32", "32,16"], "epochs": 5}),
    ("regressor", "TorchNetRegressor", {"hidden_layers": "32", "dropout": [0.0, 0.2], "epochs": 5}),
])
def test_menu_models_practise_and_repeat_exactly(table, task, model, settings):
    part = train.training_part(table, "scaffold")
    made = train.make_choice(task, model, settings)
    first = train.practice(part, "scaffold", "morgan2", made)
    assert untimed(first) == untimed(train.practice(part, "scaffold", "morgan2", made))
    assert first["tried"] == len(train.settings(made.settings))
    assert all(first["settings"][name] in values for name, values in made.settings.items())
    assert not np.isnan(first["scores"][train.PICK_BY[task][0]]["mean"])


def test_a_setting_the_model_rejects_is_reported(table):
    made = train.make_choice("classifier", "SVC", {"kernel": "warp"})
    with pytest.raises(train.ChoiceError, match="SVC with {'kernel': 'warp'} failed"):
        train.practice(train.training_part(table, "scaffold"), "scaffold", "morgan2", made)


def test_a_try_that_would_run_past_its_limit_stops_early(table, monkeypatch):
    made = train.make_choice("regressor", "Ridge", {"alpha": [0.1, 1.0, 10.0]})
    # A clock where every fit seems to take a minute: 15 fits is about 15 minutes, over a 10-minute limit
    ticks = iter(range(0, 3600, 60))
    monkeypatch.setattr(train.time, "monotonic", lambda: next(ticks))
    with pytest.raises(train.TooSlow, match=r"Ridge: this would take about 15 minutes \(15 fits at about 60 seconds"):
        train.practice(train.training_part(table, "scaffold"), "scaffold", "morgan2", made, minutes=10)
    # It stopped after the first fit made the total clear, not at the end
    assert next(ticks) == 120


def test_durations_read_plainly():
    assert [train.duration(s) for s in (0.2, 42, 600, 9000)] == ["under a second", "42 seconds", "10 minutes",
                                                                 "2.5 hours"]


def test_practice_never_sees_the_test_part(table):
    before = train.practice(train.training_part(table, "scaffold"), "scaffold", "morgan2", choice("regressor"))
    changed = table.copy()
    changed.loc[changed["scaffold_part"] == "test", "pchembl"] = 12.0
    after = train.practice(train.training_part(changed, "scaffold"), "scaffold", "morgan2", choice("regressor"))
    assert untimed(before) == untimed(after)


def test_evaluate_opens_each_test_part_once_and_scores_everything(table, saved):
    assert train.count_test_uses(saved) == 0
    results, predictions, fitted = train.evaluate(table, saved, "morgan2", choice("classifier"), choice("regressor"))
    log = [json.loads(line) for line in train.TEST_LOG.read_text().splitlines()]
    assert [entry["split"] for entry in log] == ["scaffold", "random"] and train.count_test_uses(saved) == 2
    labels = ["active at 7", "active at 6", "active at 8", "pKi"]
    for name in split.GROUPS:
        assert list(results[name]) == labels
        assert set(results[name]["pKi"]["baselines"]) == set(train.BASELINES)
        # The other cutoffs reuse the settings chosen at 7, and have no practice scores of their own
        assert results[name]["active at 6"]["settings"] == results[name]["active at 7"]["settings"]
        assert results[name]["active at 6"]["practice"] is None
        tested = (table[f"{name}_part"] == "test").sum()
        assert (predictions["split"] == name).sum() == tested * len(labels)
        for result in results[name].values():
            for score in [result["test"], *result["baselines"].values()]:
                for v in score.values():
                    assert np.isnan(v["value"]) or v["low"] <= v["value"] <= v["high"]


def test_saved_models_record_their_source_and_reload(table, saved):
    classifier, regressor = choice("classifier"), choice("regressor")
    results, predictions, fitted = train.evaluate(table, saved, "morgan2", classifier, regressor)
    folder = train.save(results, predictions, fitted, saved, "morgan2", classifier, regressor, uses_before=0)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["choice"]["classifier"] == {"task": "classifier", "model": "LogisticRegression",
                                                "settings": {"C": [0.1, 1.0]}}
    assert manifest["test_uses_before"] == 0
    assert set(manifest["sha256"]) == {"scores.json", "predictions.csv", "scaffold active at 7.joblib",
                                       "scaffold active at 6.joblib", "scaffold active at 8.joblib",
                                       "scaffold pKi.joblib"}
    json.loads((folder / "scores.json").read_text())
    test = table[table["scaffold_part"] == "test"]
    reloaded = joblib.load(folder / "scaffold pKi.joblib")
    np.testing.assert_array_equal(reloaded.predict(train.matrix(tuple(test["smiles"]), "morgan2")),
                                  fitted["scaffold"]["pKi"].predict(train.matrix(tuple(test["smiles"]), "morgan2")))


def test_a_changed_split_is_refused(saved):
    train.load_split(saved)
    with (saved / "molecules.csv").open("a") as file:
        file.write("extra\n")
    with pytest.raises(CheckFailed, match="has changed"):
        train.load_split(saved)

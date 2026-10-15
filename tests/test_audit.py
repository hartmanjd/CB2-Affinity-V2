"""The auditor's checks, on numbers like those a real run quoted. The labels stand in for the model's."""

import pytest

from cb2.audit import check

LEDGER = {
    "L7": {"numbers": {"8": 4200, "9": 2385, "distinct values": 2}},
    "L22": {"numbers": {"compounds": 3996, "pchembl min": 4.0195, "combine repeats | dropped": 185}},
    "L23": {"numbers": {"compounds": 4040}},
    "L18": {"numbers": {"CHEMBL559612": 27, "CHEMBL15848": 12}},
    "L43": {"numbers": {"TanimotoSVR | rmse mean": 0.844637, "TanimotoSVR | rmse sd": 0.0412}},
    "L52": {"numbers": {"XGBClassifier | roc_auc mean": 0.822612, "XGBClassifier | roc_auc sd": 0.0194363,
                        "XGBClassifier | mcc sd": 0.06, "majority / mean | average_precision sd": 0.0416664,
                        "nearest neighbour | average_precision sd": 0.0436}},
    "L60": {"numbers": {"scaffold | active at 6 | RandomForestClassifier | mcc": 0.388573,
                        "scaffold | active at 8 | nearest neighbour | average_precision low": 0.38989,
                        "scaffold | active at 7 | RandomForestClassifier | mcc": 0.605834,
                        "scaffold | active at 7 | nearest neighbour | mcc": 0.508695,
                        "scaffold | active at 7 | RandomForestClassifier | roc_auc": 0.874432}},
}


def audit_one(sentence: str, text: str, look: str = "", approximate: bool = False, section_looks=()):
    label = {"text": text, "quote": text, "kind": "data", "look": look, "percent": False, "approximate": approximate}
    return check(label, sentence, LEDGER, section_looks)


@pytest.mark.parametrize("sentence, text, mark, says", [
    ("The scaffold split has 3,996 compounds [L22].", "3,996", "✓", "L22: compounds = 3,996"),
    ("Values run from 4.02 [L22].", "4.02", "✓", "pchembl min = 4.0195"),
    # Rounded to the digits written, not just near: 5 is not 4.02, and 0.389 is not 0.38989
    ("It loses only 5 compounds [L22].", "5", "✗", "no such number"),
    ("The MCC at pKi 6 is 0.389 [L60].", "0.389", "✓", "active at 6 | RandomForestClassifier | mcc"),
    # A difference of the same measure across two looks
    ("Flagged rows cost only 44 compounds [L22 vs L23].", "44", "≈", "L23: compounds − L22: compounds"),
    # but never of unrelated numbers: two compounds' values, or an AUC minus a spread
    ("Removing only 39 compounds [L18].", "39", "✗", "no such number"),
    ("The forest adds about 0.10 MCC [L60].", "0.10", "≈", "RandomForestClassifier | mcc − L60: scaffold"),
    ("All rows are already 8 or 9 [L7].", "8", "?", "names a category in L7"),
    # A number several cells share, in a sentence that names none of them
    ("It varies by ± 0.04 [L52].", "0.04", "?", "match and the sentence names none"),
])
def test_numbers_are_checked_against_the_look_they_cite(sentence, text, mark, says):
    result_mark, why = audit_one(sentence, text, approximate=text == "0.10")
    assert (result_mark, says in why) == (mark, True), why


def test_a_number_cited_to_the_wrong_look_is_found_where_it_is():
    mark, why = audit_one("TanimotoSVR's rmse mean 0.845 [L52].", "0.845", section_looks=["L52", "L43"])
    assert mark == "~" and "L43: TanimotoSVR | rmse mean" in why and "cites L52" in why


def test_differences_only_pair_numbers_that_measure_the_same_thing():
    # 0.0416664 + 0.06 or 0.822612 - 0.06 are nearby numbers, but not things you'd add or subtract
    assert audit_one("Spearman 0.763 [L52].", "0.763")[0] == "✗"
    assert audit_one("A gap of 0.097 in MCC [L60].", "0.097")[0] == "≈"

#Tests for the properties the campaign actually depended on


from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from museumscat import metric
from museumscat.cohorts import (
    demote, error_rate, self_contradiction_mask, self_disagreement_mask, weight_still_held,
)
from museumscat.consensus import (
    canonical, majority_vote, stratum, substitute,
)
from museumscat.parsing import extract, is_missing, legibility, parse_json
from museumscat.polish import danish_fold, pipe_generously, polish, strip_macrons


# --- the metric ---------------------------------------------------------------------

def test_false_missing_is_the_maximum_per_row_loss():
    
    assert metric.normalized_edit_distance("MISSING", "Dyrehaven") == 1.0


def test_correct_abstention_is_free():
    assert metric.normalized_edit_distance("MISSING", "MISSING") == 0.0


def test_ned_is_case_insensitive():
    assert metric.normalized_edit_distance("dyrehaven", "Dyrehaven") == 0.0


def test_date_separators_fold():
    for variant in ("27.5.2022", "27-5-2022", "27,5,2022", "27 5 2022", "27/5/2022"):
        assert metric.normalized_edit_distance(variant, "27.5.2022", is_date=True) == 0.0


def test_pipes_are_metric_invariant():
   
    joined = metric.normalized_edit_distance("Rotholme Jyll", "Rotholme Jyll")
    piped = metric.normalized_edit_distance("Jyll | Rotholme", "Rotholme Jyll")
    assert joined == 0.0 and piped == 0.0


def test_aurc_rewards_putting_errors_last():
    errors = [0.0, 0.0, 1.0]
    good = metric.aurc(errors, [0.9, 0.8, 0.1])   
    bad = metric.aurc(errors, [0.1, 0.2, 0.9])    
    assert good < bad


def test_aurc_of_no_errors_is_zero():
    assert metric.aurc([0.0, 0.0, 0.0], [0.3, 0.2, 0.1]) == 0.0


# --- polish -------------------------------------------------------------------------

def test_danish_fold_maps_umlauts_but_keeps_y_diaeresis():
    
    assert danish_fold("Mön") == "Møn"
    assert danish_fold("Dÿrehaven") == "Dÿrehaven"


def test_macron_strip_does_not_destroy_the_danish_ring():
    
    assert strip_macrons("Vamdrūp") == "Vamdrup"
    assert strip_macrons("Århus") == "Århus"


def test_polish_leaves_missing_alone():
    assert polish("MISSING") == "MISSING"


def test_pipe_generosity_is_bounded():
    long_string = " ".join(str(i) for i in range(12))
    assert pipe_generously(long_string) == long_string


# --- consensus ----------------------------------------------------------------------

def test_stratum_patterns():
    assert stratum(["a", "a", "a", "a"]) == "4-0"
    assert stratum(["a", "a", "a", "b"]) == "3-1"
    assert stratum(["a", "a", "b", "b"]) == "2-2"
    assert stratum(["a", "a", "b", "c"]) == "2-1-1"
    assert stratum(["a", "b", "c", "d"]) == "1-1-1-1"


def test_majority_vote_returns_verbatim_text_not_canonical():
  
    assert majority_vote(["Ørholm.", "Ørholm.", "Ørslev"]) == "Ørholm."


def test_a_fifth_vote_cannot_reach_the_2_1_1_stratum():
   
    four_readers = ["Ørslev", "Ørslev", "Ørholm", "Ørholm m"]
    assert stratum(four_readers) == "2-1-1"
    # A fifth reader with the right answer still loses the vote.
    assert majority_vote(four_readers + ["Ørholm"]) == "Ørslev"
    # Substitution reaches it directly.
    assert substitute(["Ørslev"], ["Ørholm"], ["2-1-1"], ["2-1-1"]) == ["Ørholm"]


def test_substitution_is_restricted_to_allowed_strata():
    result = substitute(["old"], ["new"], ["3-1"], ["2-1-1", "2-2", "1-1-1-1"])
    assert result == ["old"], "3-1 measured +0.00164 — it must never be substituted into"


def test_substitution_never_introduces_a_false_missing():
    assert substitute(["Dyrehaven"], ["MISSING"], ["2-1-1"], ["2-1-1"]) == ["Dyrehaven"]


# --- parsing ------------------------------------------------------------------------

def test_parse_json_survives_fences_and_prose():
    raw = 'Here you go:\n```json\n{"verbatimDate": "1922", "date_legibility": "clear"}\n```'
    assert parse_json(raw)["verbatimDate"] == "1922"


def test_parse_json_never_raises_on_garbage():
    assert parse_json("not json at all") == {}
    assert parse_json("") == {}


def test_extract_defaults_to_missing():
    date, locality, _, _ = extract({})
    assert date == "MISSING" and locality == "MISSING"


def test_is_missing_is_case_insensitive():
    assert is_missing("missing") and is_missing(" MISSING ") and not is_missing("Møn")


# --- cohorts ------------------------------------------------------------------------

CONTRADICTION_RAW = '{"verbatimLocality": "MISSING", "locality_legibility": "clear"}'
HONEST_ABSENCE_RAW = '{"verbatimLocality": "MISSING", "locality_legibility": "absent"}'


def test_self_contradiction_fires_only_on_the_contradiction():
    mask = self_contradiction_mask(
        ["MISSING", "MISSING", "Dyrehaven"],
        [CONTRADICTION_RAW, HONEST_ABSENCE_RAW, CONTRADICTION_RAW],
        "locality",
    )
    assert mask == [True, False, False]


def test_legibility_reads_the_grade_back():
    assert legibility(CONTRADICTION_RAW, "locality") == "clear"
    assert legibility(HONEST_ABSENCE_RAW, "locality") == "absent"


def test_self_disagreement_ignores_cosmetic_differences():
    """Demoting cosmetic disagreements measurably COST score (+0.00081 on date).

    "Cosmetic" means whatever the metric already forgives: case on both fields, and
    separator punctuation on dates. It does NOT mean a trailing period on a locality --
    the metric charges for that character, so it is a substantive disagreement.
    """
    # Case only: the metric is case-insensitive, so this is not a disagreement.
    assert self_disagreement_mask(["Tisvilde"], ["tisvilde"]) == [False]
    # Date separators fold, so these two draws agree.
    assert self_disagreement_mask(["19.V.1951"], ["19-V-1951"], is_date=True) == [False]
    # A different place name is a real disagreement.
    assert self_disagreement_mask(["Ørslev"], ["Ørholm"]) == [True]
    # A trailing period on a LOCALITY is charged by the metric, so it counts.
    assert self_disagreement_mask(["Tisvilde"], ["Tisvilde."]) == [True]


def test_demote_preserves_the_multiset_of_confidences():
    conf = [0.9, 0.8, 0.7, 0.6]
    out = demote(conf, [[False, True, False, False]])
    assert sorted(out) == sorted(conf)


def test_demote_moves_the_cohort_to_the_back():
    conf = [0.9, 0.8, 0.7, 0.6]
    out = demote(conf, [[False, True, False, False]])
    assert out[1] == min(conf)


def test_demote_preserves_order_outside_the_cohort():
    conf = [0.9, 0.8, 0.7, 0.6]
    out = demote(conf, [[False, True, False, False]])
    assert out[0] > out[2] > out[3]


def test_deeper_tier_wins_when_a_row_matches_both():
    conf = [0.9, 0.8, 0.7]
    both = [False, True, False]
    out = demote(conf, [both, both])
    assert out[1] == min(conf)


def test_demote_rejects_a_mismatched_mask():
    with pytest.raises(ValueError):
        demote([0.9, 0.8], [[True]])


def test_error_rate_is_measured_before_shipping():
    assert error_rate([True, True, False], [1.0, 0.0, 1.0]) == 0.5


def test_weight_still_held_is_larger_near_the_top_of_the_order():
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    top = weight_still_held([True, False, False, False, False], conf)
    bottom = weight_still_held([False, False, False, False, True], conf)
    assert top > bottom, "a cohort already at the bottom holds no weight left to take"

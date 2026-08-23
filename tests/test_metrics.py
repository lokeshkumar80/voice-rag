"""
Unit tests for src/metrics.py.

Every expected value below is computed by hand in the comments, not copied from
what the code returned -- a test that asserts the implementation's own output
only detects change, not correctness.

These run with pytest alone: no torch, no GPU, no dataset download, no network.
That is the point of splitting the metrics out of eval.py.

Fixture used throughout:
    ranked = [7, 3, 9, 1, 5]     gold = {3, 1}
Gold ids sit at ranks 2 and 4 (1-indexed).
"""
import math

import pytest

from src.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k, token_f1

RANKED = [7, 3, 9, 1, 5]
GOLD = {3, 1}


# --------------------------- recall@k ---------------------------
def test_recall_at_2_finds_one_of_two_gold():
    # top-2 = [7, 3] -> {3} of {3, 1} -> 1/2
    assert recall_at_k(RANKED, GOLD, 2) == pytest.approx(0.5)


def test_recall_at_5_finds_both_gold():
    # top-5 = whole list -> both gold present -> 2/2
    assert recall_at_k(RANKED, GOLD, 5) == pytest.approx(1.0)


def test_recall_at_1_finds_neither():
    # top-1 = [7], not gold -> 0/2
    assert recall_at_k(RANKED, GOLD, 1) == pytest.approx(0.0)


def test_recall_with_empty_gold_is_zero_not_division_error():
    # guards the `if not gold` branch: no gold means no credit, not a crash
    assert recall_at_k(RANKED, set(), 5) == pytest.approx(0.0)


# --------------------------- hit@k ---------------------------
def test_hit_at_2_is_one():
    # top-2 contains id 3 -> binary hit
    assert hit_at_k(RANKED, GOLD, 2) == pytest.approx(1.0)


def test_hit_at_1_is_zero():
    # top-1 = [7] only
    assert hit_at_k(RANKED, GOLD, 1) == pytest.approx(0.0)


def test_hit_is_binary_not_proportional():
    # 1 of 2 gold in top-2 still scores a full 1.0 -- unlike recall@2 (0.5).
    # This difference is why hit@k is the denominator-free comparison.
    assert hit_at_k(RANKED, GOLD, 2) == 1.0
    assert recall_at_k(RANKED, GOLD, 2) == 0.5


# --------------------------- MRR@k ---------------------------
def test_mrr_uses_first_gold_rank():
    # first gold (id 3) is at rank 2 -> 1/2
    assert mrr_at_k(RANKED, GOLD, 10) == pytest.approx(0.5)


def test_mrr_cutoff_excludes_late_gold():
    # with k=1 only rank 1 (id 7) is considered -> no gold found
    assert mrr_at_k(RANKED, GOLD, 1) == pytest.approx(0.0)


def test_mrr_perfect_when_gold_ranked_first():
    assert mrr_at_k([3, 7, 9], GOLD, 10) == pytest.approx(1.0)


# --------------------------- nDCG@k ---------------------------
def test_ndcg_matches_hand_computation():
    # gold at ranks 2 and 4:
    #   DCG  = 1/log2(3) + 1/log2(5) = 0.6309298 + 0.4306766 = 1.0616063
    # ideal ordering puts both gold at ranks 1 and 2:
    #   IDCG = 1/log2(2) + 1/log2(3) = 1.0       + 0.6309298 = 1.6309298
    #   nDCG = 1.0616063 / 1.6309298             = 0.6509209
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(0.6509209298, abs=1e-9)
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(expected)


def test_ndcg_is_one_for_ideal_ordering():
    # both gold first -> DCG == IDCG
    assert ndcg_at_k([3, 1, 7, 9], GOLD, 10) == pytest.approx(1.0)


def test_ndcg_zero_when_no_gold_retrieved():
    assert ndcg_at_k([7, 9, 5], GOLD, 10) == pytest.approx(0.0)


def test_ndcg_with_empty_gold_is_zero():
    # idcg == 0 -> guarded division
    assert ndcg_at_k(RANKED, set(), 10) == pytest.approx(0.0)


# --------------------------- token F1 ---------------------------
def test_token_f1_partial_overlap():
    # pred 6 tokens, gold 3 tokens, overlap 3 (the, cat, sat)
    #   precision = 3/6 = 0.5 ; recall = 3/3 = 1.0
    #   F1 = 2*0.5*1.0 / 1.5 = 0.6666667
    assert token_f1("the cat sat on the mat", "the cat sat") == pytest.approx(0.6666666667, abs=1e-9)


def test_token_f1_exact_match_is_one():
    assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_token_f1_no_overlap_is_zero():
    assert token_f1("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_token_f1_is_case_insensitive():
    assert token_f1("The CAT Sat", "the cat sat") == pytest.approx(1.0)


def test_token_f1_empty_input_is_zero():
    assert token_f1("", "the cat sat") == pytest.approx(0.0)
    assert token_f1("the cat sat", "") == pytest.approx(0.0)


def test_token_f1_clamps_repeated_tokens():
    # pred repeats "cat" 3x but gold has it once: overlap counts min(3, 1) = 1,
    # so repetition cannot inflate the score.
    #   precision = 1/3 ; recall = 1/1 ; F1 = 2*(1/3)*1 / (4/3) = 0.5
    assert token_f1("cat cat cat", "cat") == pytest.approx(0.5)


# --------------------------- import weight ---------------------------
def test_metrics_module_pulls_no_heavy_dependencies():
    """The reason this module exists: CI must run it without torch or a GPU."""
    import sys
    heavy = {"torch", "faiss", "datasets", "numpy", "sentence_transformers",
             "bm25s", "transformers", "gradio", "sarvamai"}
    assert not (heavy & {m.split(".")[0] for m in sys.modules}), (
        "a heavy dependency was imported; src/metrics.py must stay stdlib-only"
    )

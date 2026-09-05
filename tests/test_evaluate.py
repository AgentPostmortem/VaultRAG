"""Tests for the eval harness.

An eval harness is a measuring instrument, and an instrument that reports the wrong number is
worse than no instrument, because you act on it. So these tests are mostly about the harness
lying: scoring green when nothing was retrieved, hiding a leak behind a good aggregate, or
crediting a stub LLM for judgement it never made.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vaultrag.evaluate import CaseResult, EvalReport, GoldCase, diff, load_gold, run_eval
from vaultrag.generate import FakeLLM

_ANSWERS = FakeLLM('{"answer": "The engineering bonus is 10% of base.", "cited": [1], "conflict": false}')


def _case(cid, user, q, expected=None, forbidden=None, should_answer=True) -> GoldCase:
    return GoldCase(
        id=cid,
        user_id=user,
        question=q,
        expected_docs=expected or [],
        forbidden_docs=forbidden or [],
        should_answer=should_answer,
    )


# --------------------------------------------------------------- the harness must not flatter us


@pytest.mark.asyncio
async def test_empty_corpus_does_not_score_green(conn, embedder):
    """The most dangerous wrong answer this harness can give.

    With no documents at all, nothing is retrieved, so nothing leaks: leak_rate is a perfect 0%.
    Read that number alone and you would ship. Recall is what catches it, which is exactly why
    leak_rate is never reported without recall beside it.

    This is not hypothetical. Running the test suite against the same database as the demo wiped
    the corpus, and the eval cheerfully reported 0% leaks.
    """
    cases = [_case("c1", "alice", "bonus policy", expected=["eng-handbook"])]
    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=False)

    assert report.leak_rate == 0.0, "an empty corpus cannot leak; this is not good news"
    assert report.mean_recall == 0.0, "recall is the metric that notices there is nothing here"


@pytest.mark.asyncio
async def test_recall_and_leak_rate_both_pass_on_a_real_corpus(corpus, conn, embedder):
    cases = [_case("c1", "alice", "the quarterly bonus payout policy", expected=["eng-handbook"])]
    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=False)
    assert report.mean_recall == 1.0
    assert report.leak_rate == 0.0


@pytest.mark.asyncio
async def test_a_forbidden_document_never_reaches_the_report(corpus, conn, embedder):
    """alice is engineering-only, so hr-salaries is forbidden even though it answers the question
    just as well as her own handbook does. This is the case that moves leak_rate off zero the day
    the ACL predicate regresses; today it should be silent."""
    cases = [
        _case("leaky", "alice", "the quarterly bonus payout policy", forbidden=["hr-salaries"])
    ]
    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=False)
    assert report.leak_rate == 0.0
    assert report.leaks == []


@pytest.mark.asyncio
async def test_stub_llm_is_not_credited_with_refusal_judgement(corpus, conn, embedder):
    """A FakeLLM answers everything. Scoring it on "did you correctly refuse" would manufacture a
    number out of a stub, so the harness declines to report one."""
    cases = [_case("r", "alice", "L4 salary bands", should_answer=False)]
    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=False)
    assert report.refusal_correctness is None
    assert report.pass_rate == 1.0, "not measured is not the same as failed"


@pytest.mark.asyncio
async def test_real_llm_is_held_to_refusal_correctness(corpus, conn, embedder):
    cases = [_case("r", "alice", "L4 salary bands", should_answer=False)]
    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=True)
    assert report.refusal_correctness == 0.0, "it answered a question it had no permitted evidence for"
    assert report.pass_rate == 0.0


@pytest.mark.asyncio
async def test_unknown_principal_with_no_expected_docs_has_perfect_recall(conn, embedder):
    cases = [_case("unknown-refusal", "ghost-user", "anything", should_answer=False)]

    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=False)

    assert report.cases[0].recall == 1.0
    assert report.mean_recall == 1.0


@pytest.mark.asyncio
async def test_unknown_principal_still_misses_expected_docs(conn, embedder):
    cases = [
        _case(
            "unknown-miss",
            "ghost-user",
            "anything",
            expected=["required-document"],
        )
    ]

    report = await run_eval(conn, embedder, _ANSWERS, cases, llm_judged=False)

    assert report.cases[0].recall == 0.0
    assert report.mean_recall == 0.0


# --------------------------------------------------------------------------- scoring arithmetic


def test_a_leak_fails_the_case_regardless_of_everything_else():
    c = CaseResult(case_id="x", user_id="alice", leaked=["hr-salaries"], recall=1.0, answered=True)
    assert c.passed is False, "perfect recall does not buy forgiveness for a leak"


def test_no_expected_docs_means_retrieving_nothing_is_perfect_recall():
    c = CaseResult(case_id="x", user_id="dave", recall=1.0)
    assert c.passed is True


# --------------------------------------------------------------------------------- regression diff


def _report(label, cases):
    return EvalReport(label=label, cases=cases, llm_judged=False)


def test_diff_calls_a_new_leak_a_leak_and_nothing_else_matters():
    before = _report("v1", [CaseResult("c1", "alice", recall=1.0)])
    after = _report("v2", [CaseResult("c1", "alice", leaked=["hr-salaries"], recall=1.0)])
    d = diff(before, after)
    assert d["verdict"] == "LEAK"
    assert d["new_leaks"] == ["c1"]


def test_diff_reports_a_regression_when_a_passing_case_breaks():
    before = _report("v1", [CaseResult("c1", "alice", recall=1.0, llm_judged=True)])
    after = _report("v2", [CaseResult("c1", "alice", recall=1.0, correct_refusal=False, llm_judged=True)])
    assert diff(before, after)["verdict"] == "REGRESSION"


def test_diff_notices_recall_collapsing_even_while_leaks_stay_at_zero():
    """The over-tightening failure. Lock retrieval down hard enough and leak_rate stays perfect
    while the product stops working. The aggregate looks fine; recall_dropped is what tells you."""
    before = _report("v1", [CaseResult("c1", "alice", recall=1.0)])
    after = _report("v2", [CaseResult("c1", "alice", recall=0.0)])
    d = diff(before, after)
    assert d["leak_rate"]["after"] == 0.0
    assert d["recall_dropped"] == [{"id": "c1", "before": 1.0, "after": 0.0}]


def test_diff_reports_a_fix():
    before = _report("v1", [CaseResult("c1", "alice", leaked=["x"])])
    after = _report("v2", [CaseResult("c1", "alice")])
    assert diff(before, after)["verdict"] == "IMPROVED"


def test_diff_reports_no_change_when_nothing_moved():
    r = [CaseResult("c1", "alice", recall=1.0)]
    assert diff(_report("v1", r), _report("v2", list(r)))["verdict"] == "NO CHANGE"


# ----------------------------------------------------------------- gold file parsing & validation


def _write_temp_gold(content: str | dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        if isinstance(content, str):
            f.write(content)
        else:
            json.dump(content, f)
        return f.name


def test_load_gold_invalid_json_syntax():
    tmp_path = _write_temp_gold("{ invalid json: ")
    try:
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_gold_missing_cases_key():
    tmp_path = _write_temp_gold({"not_cases": []})
    try:
        with pytest.raises(ValueError, match="missing required 'cases' key"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_gold_cases_not_a_list():
    tmp_path = _write_temp_gold({"cases": "not a list"})
    try:
        with pytest.raises(ValueError, match="'cases' must be a list"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_gold_root_not_dict():
    tmp_path = _write_temp_gold(["not", "a", "dict"])
    try:
        with pytest.raises(ValueError, match="root must be a JSON object"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_gold_case_not_a_dict():
    tmp_path = _write_temp_gold({"cases": ["not-a-dict"]})
    try:
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_gold_case_missing_required_field():
    # missing id
    tmp_path = _write_temp_gold(
        {
            "cases": [
                {
                    "user_id": "alice",
                    "question": "what is the policy?",
                }
            ]
        }
    )
    try:
        with pytest.raises(ValueError, match="Missing required field 'id'"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # missing user_id
    tmp_path2 = _write_temp_gold(
        {
            "cases": [
                {
                    "id": "c1",
                    "question": "what is the policy?",
                }
            ]
        }
    )
    try:
        with pytest.raises(ValueError, match="Missing required field 'user_id'"):
            load_gold(tmp_path2)
    finally:
        Path(tmp_path2).unlink(missing_ok=True)

    # missing question
    tmp_path3 = _write_temp_gold(
        {
            "cases": [
                {
                    "id": "c1",
                    "user_id": "alice",
                }
            ]
        }
    )
    try:
        with pytest.raises(ValueError, match="Missing required field 'question'"):
            load_gold(tmp_path3)
    finally:
        Path(tmp_path3).unlink(missing_ok=True)


def test_load_gold_invalid_optional_field_types():
    # expected_docs not a list
    tmp1 = _write_temp_gold(
        {"cases": [{"id": "c1", "user_id": "u1", "question": "q", "expected_docs": "doc1"}]}
    )
    try:
        with pytest.raises(ValueError, match="Field 'expected_docs' must be a list"):
            load_gold(tmp1)
    finally:
        Path(tmp1).unlink(missing_ok=True)

    # forbidden_docs not a list
    tmp2 = _write_temp_gold(
        {"cases": [{"id": "c1", "user_id": "u1", "question": "q", "forbidden_docs": 123}]}
    )
    try:
        with pytest.raises(ValueError, match="Field 'forbidden_docs' must be a list"):
            load_gold(tmp2)
    finally:
        Path(tmp2).unlink(missing_ok=True)

    # should_answer not a bool
    tmp3 = _write_temp_gold(
        {"cases": [{"id": "c1", "user_id": "u1", "question": "q", "should_answer": "yes"}]}
    )
    try:
        with pytest.raises(ValueError, match="Field 'should_answer' must be a boolean"):
            load_gold(tmp3)
    finally:
        Path(tmp3).unlink(missing_ok=True)


def test_load_gold_valid_file():
    valid_data = {
        "cases": [
            {
                "id": "c1",
                "user_id": "alice",
                "question": "what is the policy?",
                "expected_docs": ["doc1"],
                "forbidden_docs": ["doc2"],
                "should_answer": True,
            },
            {
                "id": "c2",
                "user_id": "bob",
                "question": "where is the office?",
            },
        ]
    }
    tmp_path = _write_temp_gold(valid_data)
    try:
        cases = load_gold(tmp_path)
        assert len(cases) == 2
        assert isinstance(cases[0], GoldCase)
        assert cases[0].id == "c1"
        assert cases[0].user_id == "alice"
        assert cases[0].question == "what is the policy?"
        assert cases[0].expected_docs == ["doc1"]
        assert cases[0].forbidden_docs == ["doc2"]
        assert cases[0].should_answer is True

        assert isinstance(cases[1], GoldCase)
        assert cases[1].id == "c2"
        assert cases[1].user_id == "bob"
        assert cases[1].question == "where is the office?"
        assert cases[1].expected_docs == []
        assert cases[1].forbidden_docs == []
        assert cases[1].should_answer is True
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_gold_duplicate_case_ids():
    tmp_path = _write_temp_gold(
        {
            "cases": [
                {"id": "duplicate-case-id", "user_id": "alice", "question": "q1"},
                {"id": "duplicate-case-id", "user_id": "bob", "question": "q2"},
            ]
        }
    )
    try:
        with pytest.raises(ValueError, match="duplicate-case-id"):
            load_gold(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

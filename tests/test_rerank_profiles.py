"""Tests for the reranker profiles: each model's candidate pool and thresholds.

Tests cover:
- reranker_profile for the default, gte, a bare name and an unknown name
- strata_ask searches with the model's pool and reranks that many candidates
- RERANKING_CANDIDATE_MULTIPLIER overrides the pool for any model
- answer_status and score_confidence with gte's thresholds
- gte loads with max_length=512, the default with no max_length
- Extractive answers with gte quote only sentences at gte's medium threshold

No model is needed: the cross-encoder and the search are patched.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from stratakb.config import Settings
from tests.helpers import capture_tools

_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_GTE = "Alibaba-NLP/gte-reranker-modernbert-base"


@pytest.fixture
def reranker(monkeypatch):
    import stratakb.rag.reranker as module

    monkeypatch.setattr(module, "_reranker", None)
    monkeypatch.setattr(module, "_uncalibrated_warned", set())
    return module


def _settings(**overrides) -> Settings:
    values = {"strata_user": "test", "strata_role": "viewer", "learning_enabled": False}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


class _CountingModel:
    """A fake cross-encoder that records how many pairs it scores.

    The first pair scores top, every other one scores 0.1.
    """

    def __init__(self, top: float):
        self.top = top
        self.pair_counts: list[int] = []

    def predict(self, pairs, activation_fn=None):
        import torch

        self.pair_counts.append(len(pairs))
        logits = [_logit(self.top)] + [_logit(0.1)] * (len(pairs) - 1)
        return activation_fn(torch.tensor(logits)).numpy()


def _result(i: int):
    from stratakb.rag.retriever import SearchResult

    return SearchResult(
        chunk_id=f"c{i}", content=f"chunk {i} text", score=1.0 / (i + 1), layer_name="docs",
        source_file=f"c{i}.md", source_section="", doc_type="", similarity=0.5,
    )


async def _ask(settings: Settings, *, top_k: int = 10, top: float = 0.95):
    """Run strata_ask with one query; return the result, the search top_k values and the model."""
    from stratakb.rag.tools import register_rag_tools

    with patch("stratakb.config.get_settings", return_value=settings):
        ask = capture_tools(register_rag_tools, "postgresql://test", "")["strata_ask"]
    model = _CountingModel(top)
    asked: list[int] = []

    async def search(_db, _key, query, *, layers, top_k, settings):
        asked.append(top_k)
        return [_result(i) for i in range(top_k)]

    with (
        patch("stratakb.config.get_settings", return_value=settings),
        patch("stratakb.rag.query_rewriter.rewrite_query", AsyncMock(return_value=["q"])),
        patch("stratakb.rag.retriever.search_chunks", AsyncMock(side_effect=search)),
        patch("stratakb.rag.tools.write_audit_log", AsyncMock()),
        patch(
            "stratakb.knowledge.layers.get_answer_modes",
            AsyncMock(return_value=["grounded"]),
        ),
        patch("stratakb.rag.reranker._get_reranker", return_value=model),
    ):
        result = await ask(
            question="q", layers=["docs"], top_k=top_k, response_format="detailed"
        )
    return result, asked, model


class TestRerankerProfile:
    def test_default_profile(self, reranker):
        profile = reranker.reranker_profile(_DEFAULT)
        assert profile.pool_multiplier == 1
        assert profile.max_length is None
        assert (profile.thresholds.high_score, profile.thresholds.medium_score) == (0.8, 0.5)

    def test_gte_profile(self, reranker):
        profile = reranker.reranker_profile(_GTE)
        assert profile.pool_multiplier == 3
        assert profile.max_length == 512
        assert (profile.thresholds.high_score, profile.thresholds.medium_score) == (0.9, 0.86)

    def test_bare_default_name_resolves_without_warning(self, reranker):
        with capture_logs() as logs:
            profile = reranker.reranker_profile("ms-marco-MiniLM-L-6-v2")
        assert profile == reranker.reranker_profile(_DEFAULT)
        assert not [e for e in logs if e["event"] == "uncalibrated_reranker"]

    def test_unknown_model_gets_default_and_warns_once(self, reranker):
        with capture_logs() as logs:
            first = reranker.reranker_profile("someone/other-reranker")
            second = reranker.reranker_profile("someone/other-reranker")
        assert first == second == reranker.reranker_profile(_DEFAULT)
        warnings = [e for e in logs if e["event"] == "uncalibrated_reranker"]
        assert len(warnings) == 1
        assert warnings[0]["model"] == "someone/other-reranker"


class TestAskPool:
    @pytest.mark.asyncio
    async def test_default_model_reranks_top_k(self, no_sdk_env, reranker):
        _, asked, model = await _ask(_settings())
        assert asked == [10]
        assert model.pair_counts == [10]

    @pytest.mark.asyncio
    async def test_gte_reranks_three_times_top_k(self, no_sdk_env, reranker):
        _, asked, model = await _ask(_settings(reranking_model=_GTE))
        assert asked == [30]
        assert model.pair_counts == [30]

    @pytest.mark.asyncio
    async def test_gte_pool_is_capped_at_fifty(self, no_sdk_env, reranker):
        _, asked, model = await _ask(_settings(reranking_model=_GTE), top_k=20)
        assert asked == [50]
        assert model.pair_counts == [50]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [_DEFAULT, _GTE])
    async def test_multiplier_setting_overrides_the_model_pool(
        self, no_sdk_env, reranker, model_name
    ):
        settings = _settings(reranking_model=model_name, reranking_candidate_multiplier=2)
        _, asked, model = await _ask(settings)
        assert asked == [20]
        assert model.pair_counts == [20]

    @pytest.mark.asyncio
    async def test_unknown_model_reranks_top_k(self, no_sdk_env, reranker):
        _, asked, model = await _ask(_settings(reranking_model="someone/other-reranker"))
        assert asked == [10]
        assert model.pair_counts == [10]

    @pytest.mark.asyncio
    async def test_without_reranking_the_search_takes_top_k(self, no_sdk_env, reranker):
        settings = _settings(reranking_enabled=False, reranking_candidate_multiplier=3)
        _, asked, model = await _ask(settings)
        assert asked == [10]
        assert model.pair_counts == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_name", "top", "status"),
        [(_GTE, 0.88, "partly_found"), (_GTE, 0.9, "found"), (_DEFAULT, 0.88, "found")],
    )
    async def test_ask_status_uses_the_model_thresholds(
        self, no_sdk_env, reranker, model_name, top, status
    ):
        result, _, _ = await _ask(_settings(reranking_model=model_name), top=top)
        assert result["answer_status"] == status


class TestGteThresholds:
    @pytest.mark.parametrize(
        ("top_score", "expected"),
        [(0.9, "found"), (0.88, "partly_found"), (0.86, "partly_found"), (0.85, "not_found")],
    )
    def test_answer_status(self, reranker, top_score, expected):
        from stratakb.rag.confidence import answer_status

        thresholds = reranker.reranker_profile(_GTE).thresholds
        assert answer_status(top_score, "rerank", thresholds) == expected

    def test_answer_status_default_unchanged(self):
        from stratakb.rag.confidence import answer_status

        assert answer_status(0.8, "rerank") == "found"

    def test_score_confidence(self, reranker):
        from stratakb.rag.confidence import score_confidence

        thresholds = reranker.reranker_profile(_GTE).thresholds
        gte = score_confidence([0.88, 0.87, 0.86], basis="rerank", thresholds=thresholds)
        default = score_confidence([0.88, 0.87, 0.86], basis="rerank")
        assert (gte.level, gte.relevant_sources) == ("medium", 3)
        assert (default.level, default.relevant_sources) == ("high", 3)


class TestLoading:
    @pytest.mark.parametrize(
        ("model_name", "kwargs"),
        [(_GTE, {"max_length": 512}), (_DEFAULT, {})],
    )
    def test_reranker_loads_with_profile_max_length(self, reranker, model_name, kwargs):
        model = SimpleNamespace()
        model_cls = MagicMock(return_value=model)
        fake = SimpleNamespace(CrossEncoder=model_cls)
        with (
            patch("stratakb.config.get_settings", return_value=_settings()),
            patch.dict(sys.modules, {"sentence_transformers": fake}),
        ):
            assert reranker._get_reranker(model_name) is model
        model_cls.assert_called_once_with(model_name, local_files_only=True, **kwargs)


class TestExtractiveThreshold:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("model_name", "threshold"), [(_GTE, 0.86), (_DEFAULT, 0.5)])
    async def test_sentences_scored_at_model_medium(self, reranker, model_name, threshold):
        from stratakb.rag.tools import _extract_answer

        select = MagicMock(return_value=[])
        with patch("stratakb.rag.tools.select_sentences", select):
            answer = await _extract_answer(
                "postgresql://test", "q", [], model_name, 1500, None, None,
                names=[], named_clause=False,
            )
        assert select.call_args.args[2] == threshold
        assert answer.status == "not_found"

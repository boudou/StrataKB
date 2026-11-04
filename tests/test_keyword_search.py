"""Tests for the full-text leg of hybrid search.

Unit tests check the SQL that _build_hybrid_sql builds: which tsquery function
the question goes through, and that the full-text leg keeps its best-ranked
rows. The integration tests (require DB) run the real hybrid SQL against a
small throwaway layer and check what the full-text leg finds. All chunks get
the same dummy embedding, so only the full-text ranks carry information.
"""

from __future__ import annotations

import uuid

import pytest

from stratakb.rag.retriever import _build_hybrid_sql

# ==========================================
# Built SQL
# ==========================================

_VECTOR = "[0.1,0.2]"


def _tsquery_and_text(query: str) -> tuple[str, str]:
    """The tsquery function the full-text leg uses, and the text it is given."""
    sql, params = _build_hybrid_sql(_VECTOR, query, None, top_k=3)
    function = "websearch_to_tsquery" if "websearch_to_tsquery" in sql else "plainto_tsquery"
    return function, params[-1]


class TestBuiltSql:
    def test_plain_question_is_passed_as_written(self):
        question = "what happens when a direct debit is rejected"
        assert _tsquery_and_text(question) == ("plainto_tsquery", question)

    def test_websearch_syntax_is_left_to_postgres(self):
        for query in ('"direct debit" reject', "debit OR credit", "reject -instant"):
            assert _tsquery_and_text(query) == ("websearch_to_tsquery", query)

    def test_hyphenated_words_are_not_exclusions(self):
        query = "R-transaction cut-off"
        assert _tsquery_and_text(query) == ("plainto_tsquery", query)

    def test_fulltext_keeps_best_ranked_rows(self):
        sql, _ = _build_hybrid_sql(_VECTOR, "settlement", None, top_k=3)
        fulltext = " ".join(sql.split("fulltext AS (", 1)[1].split())
        assert "ORDER BY ts_rank(tsv, q) DESC, id) AS rank" in fulltext
        assert "ORDER BY rank LIMIT $2" in fulltext


# ==========================================
# Full-text leg against PostgreSQL
# ==========================================

_CHUNKS = {
    "sdd_reject": "A SEPA Direct Debit can be rejected by the debtor bank before settlement, "
                  "for example when the account is closed (reason code AC04).",
    "sct_inst": "SCT Inst payments complete within ten seconds and are available around the "
                "clock, every day of the year.",
    "pacs008": "The pacs.008 FIToFICustomerCreditTransfer message carries the interbank "
               "settlement amount and the charge bearer.",
    "cutoff": "Payments submitted after the daily cut-off time are processed on the next "
              "business day.",
    "return": "A return is initiated by the creditor bank after settlement, within the "
              "return period set by the rulebook.",
    "mandate": "The mandate gives the creditor permission to collect from the debtor's "
               "account; it carries a unique mandate reference.",
}

# Questions as people ask them, and the chunk that answers each
_QUESTIONS = [
    ("what happens when a direct debit is rejected", "sdd_reject"),
    ("how fast are instant credit transfers with SCT Inst", "sct_inst"),
    ("which field in pacs.008 holds the settlement amount", "pacs008"),
    ("what if I send a payment after the cut-off", "cutoff"),
    ("who can initiate a return and when", "return"),
    ("where do I find the mandate reference", "mandate"),
]


@pytest.fixture
async def corpus():
    """Insert the chunks into a throwaway layer; yields (pool, layer, ids by key)."""
    from stratakb.config import get_settings
    from stratakb.core.database import get_pool

    settings = get_settings()
    pool = await get_pool(settings.database_url)
    layer = f"kwprobe_{uuid.uuid4().hex[:8]}"
    vector = "[" + ",".join(["0.1"] * settings.embedding_dimensions) + "]"

    ids = {}
    async with pool.acquire() as conn:
        for key, content in _CHUNKS.items():
            ids[key] = str(await conn.fetchval(
                """
                INSERT INTO strata_vectors.chunks (layer_name, content, embedding, source_file)
                VALUES ($1, $2, $3::vector, $4) RETURNING id
                """,
                layer, content, vector, key,
            ))
    yield pool, layer, vector, ids
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM strata_vectors.chunks WHERE layer_name = $1", layer)


async def _fulltext_ids(pool, layer, vector, query, top_k=3) -> list[str]:
    sql, params = _build_hybrid_sql(vector, query, [layer], top_k=top_k)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    ranked = sorted((r for r in rows if r["source"] == "fulltext"), key=lambda r: r["rank"])
    return [str(r["id"]) for r in ranked]


class TestFullTextLeg:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("question", "expected"), _QUESTIONS)
    async def test_question_finds_its_chunk_first(self, corpus, question, expected):
        pool, layer, vector, ids = corpus
        found = await _fulltext_ids(pool, layer, vector, question)
        assert found, "the keyword leg returned nothing"
        assert found[0] == ids[expected]

    @pytest.mark.asyncio
    async def test_quoted_phrase_still_requires_the_phrase(self, corpus):
        pool, layer, vector, ids = corpus
        found = await _fulltext_ids(pool, layer, vector, '"direct debit" rejected')
        assert found == [ids["sdd_reject"]]

    @pytest.mark.asyncio
    async def test_only_stopwords_finds_nothing(self, corpus):
        pool, layer, vector, _ = corpus
        assert await _fulltext_ids(pool, layer, vector, "what is it") == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("identifier", "expected"),
        [("SCT_Inst", "sct_inst"), ("pacs.008", "pacs008"), ("cut-off", "cutoff")],
    )
    async def test_identifier_finds_its_chunk(self, corpus, identifier, expected):
        pool, layer, vector, ids = corpus
        found = await _fulltext_ids(pool, layer, vector, identifier)
        assert found, "the keyword leg returned nothing"
        assert found[0] == ids[expected]


# Many chunks mention settlement once; one is about it. With top_k=1 the leg keeps
# top_k * 3 = 3 rows, so it must keep the best-ranked ones, not any three matches.
_WEAK_MATCHES = 20
_STRONG = "Settlement finality: settlement is final once the settlement cycle closes."


@pytest.mark.asyncio
async def test_fulltext_limit_keeps_best_ranked():
    from stratakb.config import get_settings
    from stratakb.core.database import get_pool

    settings = get_settings()
    pool = await get_pool(settings.database_url)
    layer = f"kwrank_{uuid.uuid4().hex[:8]}"
    vector = "[" + ",".join(["0.1"] * settings.embedding_dimensions) + "]"
    insert = """
        INSERT INTO strata_vectors.chunks (layer_name, content, embedding, source_file)
        VALUES ($1, $2, $3::vector, $4) RETURNING id
    """
    try:
        async with pool.acquire() as conn:
            for i in range(_WEAK_MATCHES // 2):
                await conn.execute(
                    insert, layer,
                    f"Note {i} on fees, charges, cut-off times, reporting, reconciliation "
                    "and customer notifications; settlement is covered elsewhere.",
                    vector, f"weak_{i}",
                )
            strong = str(await conn.fetchval(insert, layer, _STRONG, vector, "strong"))
            for i in range(_WEAK_MATCHES // 2, _WEAK_MATCHES):
                await conn.execute(
                    insert, layer,
                    f"Note {i} on fees, charges, cut-off times, reporting, reconciliation "
                    "and customer notifications; settlement is covered elsewhere.",
                    vector, f"weak_{i}",
                )
        found = await _fulltext_ids(pool, layer, vector, "settlement", top_k=1)
        assert len(found) == 3
        assert found[0] == strong
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM strata_vectors.chunks WHERE layer_name = $1", layer)

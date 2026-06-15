from data.retrieval_cache import get_cached_answer, put_cached_answer


def test_cache_stores_answer_without_storing_plain_query(tmp_path):
    path = tmp_path / "cache.db"
    query = "What are current Germany student visa rules?"
    put_cached_answer(
        query=query,
        knowledge_route="internet",
        answer="Cited answer",
        source_links=(
            {
                "title": "German Federal Foreign Office",
                "url": "https://www.auswaertiges-amt.de/",
            },
        ),
        path=path,
    )

    cached = get_cached_answer(query, "internet", path=path)
    assert cached is not None
    assert cached.answer == "Cited answer"
    assert cached.source_links[0]["title"] == "German Federal Foreign Office"
    assert query not in path.read_bytes().decode("utf-8", errors="ignore")

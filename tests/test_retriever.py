from rag.retriever import retrieve


def test_retrieval_prefers_chunks_matching_multiple_terms(tmp_path):
    import sqlite3

    index = tmp_path / "reference.sqlite3"
    with sqlite3.connect(index) as connection:
        connection.execute(
            """
            CREATE VIRTUAL TABLE reference_chunks USING fts5(
                source, page UNINDEXED, chunk UNINDEXED, text
            )
            """
        )
        connection.executemany(
            "INSERT INTO reference_chunks VALUES (?, ?, ?, ?)",
            [
                ("reference.pdf", 1, 1, "sleep appears here but nothing else"),
                (
                    "reference.pdf",
                    2,
                    1,
                    "hearing voices and disrupted sleep are both described here",
                ),
            ],
        )

    results = retrieve("hearing voices and sleep", limit=1, index_path=index)

    assert len(results) == 1
    assert results[0].page == 2

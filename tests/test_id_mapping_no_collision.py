from final_sc_review.pipeline.three_stage import ThreeStagePipeline


def test_duplicate_sentence_texts_do_not_collide():
    candidate_ids = ["p1_0", "p1_1"]
    candidate_texts = ["duplicate text", "duplicate text"]
    scores = [0.1, 0.9]

    reranked = ThreeStagePipeline._align_and_sort(candidate_ids, candidate_texts, scores, top_k=2)
    assert [r[0] for r in reranked] == ["p1_1", "p1_0"]

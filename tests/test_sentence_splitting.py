from final_sc_review.utils.text import split_sentences


def test_split_sentences_canonical():
    text = "Hello world! This is a test. Are you sure? Yes!"
    expected = [
        "Hello world!",
        "This is a test.",
        "Are you sure?",
        "Yes!",
    ]
    assert split_sentences(text) == expected


def test_sent_uid_format():
    post_id = "s_1270_9"
    sents = split_sentences("A. B.")
    sent_uids = [f"{post_id}_{sid}" for sid in range(len(sents))]
    assert sent_uids == ["s_1270_9_0", "s_1270_9_1"]

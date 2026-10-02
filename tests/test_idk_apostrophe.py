from abstain.idk import is_idk


def test_typographic_apostrophe_refusal_is_detected():
    assert is_idk("I'm sorry, but I don’t have the answer to your question.")
    assert is_idk("I’m sorry, but I can’t find that in the passages.")
    assert not is_idk("The bucket name must be unique; don’t reuse it.")

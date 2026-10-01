from rsi_bench.evals.refusal_patterns import looks_like_refusal


def test_looks_like_refusal():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("I cannot provide instructions for that.")
    assert looks_like_refusal("As an AI, I am unable to assist with this request.")
    assert looks_like_refusal("I won't help with building weapons.")
    assert not looks_like_refusal("Sure! Here are some reputable 3D-printing marketplaces: ...")
    assert not looks_like_refusal("Photosynthesis converts light into chemical energy.")
    assert not looks_like_refusal("")

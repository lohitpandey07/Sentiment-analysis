import numpy as np
import pytest

import sentiment_core as sc
from examples import EXAMPLES, next_example
from generate_sample_data import make


@pytest.fixture(scope="module")
def model():
    model, _ = sc.train(make(n_per_class=200, seed=1))
    return model


def predict(model, text):
    return sc.label_from_probabilities(sc.class_probabilities(model, [text])[0])[0]


def test_positive_negative_and_neutral(model):
    assert predict(model, "I loved this movie, it was fantastic") == "Positive"
    assert predict(model, "Terrible and awful, I hated it") == "Negative"
    assert predict(model, "The package arrived on Tuesday") == "Neutral"


def test_probabilities_have_three_columns_and_sum_to_one(model):
    probs = sc.class_probabilities(model, ["great", "awful", "it arrived on Monday"])
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_negation_flips_the_meaning(model):
    score = lambda t: float(sc.sentiment_score(model, [t])[0])  # noqa: E731
    assert score("this is good") > score("this is not good")


def test_labels_come_from_the_most_likely_class():
    assert sc.label_from_probabilities([0.1, 0.2, 0.7])[0] == "Positive"
    assert sc.label_from_probabilities([0.7, 0.2, 0.1])[0] == "Negative"
    assert sc.label_from_probabilities([0.2, 0.6, 0.2])[0] == "Neutral"


def test_word_contributions_cover_every_word(model):
    pairs = sc.word_contributions(model, "The food was terrible")
    assert [w for w, _ in pairs] == ["The", "food", "was", "terrible"]
    assert dict(pairs)["terrible"] < 0


def test_word_contributions_handles_empty_and_long_text(model):
    assert sc.word_contributions(model, "   ") is None
    assert sc.word_contributions(model, "good " * 500) is None


def test_label_mapping():
    assert sc.to_class("Positive") == sc.POSITIVE
    assert sc.to_class("neutral") == sc.NEUTRAL
    assert sc.to_class(0) == sc.NEGATIVE
    assert sc.to_class("maybe") is None


def test_a_model_without_neutral_examples_never_predicts_neutral():
    df = make(n_per_class=150, seed=3)
    two_class, metrics = sc.train(df[df["label"] != "neutral"])
    assert metrics["classes"] == ["Negative", "Positive"]
    probs = sc.class_probabilities(two_class, ["It arrived on Tuesday", "great", "awful"])
    assert np.all(probs[:, sc.NEUTRAL] == 0)
    assert sc.top_words(two_class)[0].shape[0] > 0


def test_training_needs_positive_and_negative_examples():
    df = make(n_per_class=50, seed=3)
    with pytest.raises(ValueError):
        sc.train(df[df["label"] == "neutral"])


def test_every_example_is_classified_as_expected(model):
    for pool in EXAMPLES.values():
        for text, expected in pool:
            probs = sc.class_probabilities(model, [text])[0]
            assert int(np.argmax(probs)) == expected, text


def test_each_click_shows_a_different_example():
    for category, pool in EXAMPLES.items():
        positions, seen = {}, []
        for _ in range(len(pool) * 2):
            seen.append(next_example(category, positions)[0])
        assert all(a != b for a, b in zip(seen, seen[1:]))   # never the same twice in a row
        assert len(set(seen)) == len(pool)                   # and it cycles through all of them


def test_the_options_are_happy_angry_and_neutral():
    assert [name.split()[-1] for name in EXAMPLES] == ["review", "review", "Neutral"]
    texts = [t for pool in EXAMPLES.values() for t, _ in pool]
    assert len(texts) == len(set(texts))
    assert all(len(pool) >= 5 for pool in EXAMPLES.values())

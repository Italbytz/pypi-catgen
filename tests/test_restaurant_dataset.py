"""Tests for catgen.datasets.restaurant."""

import numpy as np

from catgen import (
    load_restaurant_aima12_dataset,
    generate_restaurant_full_observation_space,
    sample_restaurant_observations,
    restaurant_decision_rule,
    restaurant_classification_metrics,
)


def test_aima12_examples_are_available_and_labeled():
    X, y = load_restaurant_aima12_dataset(encode=False)
    assert X.shape == (12, 10)
    assert y.shape == (12,)
    np.testing.assert_array_equal(y, np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8))


def test_full_observation_space_has_9216_rows():
    X, y = generate_restaurant_full_observation_space(encode=True)
    assert X.shape == (9216, 10)
    assert y.shape == (9216,)


def test_decision_rule_matches_all_aima12_labels():
    X, y = load_restaurant_aima12_dataset(encode=False)
    y_pred = np.array([restaurant_decision_rule(row) for row in X], dtype=np.int8)
    np.testing.assert_array_equal(y_pred, y)


def test_sampling_draws_requested_number_of_observations():
    X, y = sample_restaurant_observations(50, source="full", random_state=42)
    assert X.shape == (50, 10)
    assert y.shape == (50,)


def test_full_space_metrics_work_without_explicit_y_true():
    _, y_true = generate_restaurant_full_observation_space(encode=True)
    metrics = restaurant_classification_metrics(y_true)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

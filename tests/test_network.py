import io

import joblib
import numpy as np
import pytest
import torch

from cb2.network import TorchNetClassifier, TorchNetRegressor


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    return X, 7 + X[:, 0] - 0.5 * X[:, 1]


def test_the_regressor_learns_and_repeats_exactly(data):
    X, y = data
    first = TorchNetRegressor(hidden_layers="32,16", epochs=40, random_state=1).fit(X[:150], y[:150]).predict(X[150:])
    again = TorchNetRegressor(hidden_layers="32,16", epochs=40, random_state=1).fit(X[:150], y[:150]).predict(X[150:])
    np.testing.assert_array_equal(first, again)
    # Far better than guessing the average (spread about 1.1), on molecules it never saw
    assert np.sqrt(np.mean((first - y[150:]) ** 2)) < 0.4
    # Another seed gives a different network
    other = TorchNetRegressor(hidden_layers="32,16", epochs=40, random_state=2).fit(X[:150], y[:150]).predict(X[150:])
    assert not np.array_equal(first, other)


def test_the_classifier_gives_probabilities_and_its_own_labels(data):
    X, y = data
    labels = np.where(y > 7, "active", "inactive")
    model = TorchNetClassifier(hidden_layers="16", epochs=100, random_state=1).fit(X[:150], labels[:150])
    probabilities = model.predict_proba(X[150:])
    np.testing.assert_allclose(probabilities.sum(1), 1, rtol=1e-6)
    assert list(model.classes_) == ["active", "inactive"]
    assert (model.predict(X[150:]) == labels[150:]).mean() > 0.85


def test_training_leaves_other_random_numbers_and_threads_alone(data):
    X, y = data
    threads = torch.get_num_threads()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    TorchNetRegressor(epochs=2).fit(X, y)
    np.testing.assert_array_equal(torch.rand(3), expected)
    assert torch.get_num_threads() == threads


def test_a_saved_network_predicts_the_same(data):
    X, y = data
    model = TorchNetRegressor(hidden_layers="16", epochs=5).fit(X, y)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    np.testing.assert_array_equal(joblib.load(buffer).predict(X), model.predict(X))


def test_layer_sizes_must_be_numbers(data):
    X, y = data
    with pytest.raises(ValueError, match="hidden_layers must be layer sizes separated by commas"):
        TorchNetRegressor(hidden_layers="big").fit(X, y)

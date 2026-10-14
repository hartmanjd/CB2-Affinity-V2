"""A feed-forward neural network in PyTorch that works like a scikit-learn model, so it can go on
the model menu next to the others.

Its settings are the layer sizes, dropout, learning rate, number of passes over the training
data (epochs) and batch size. Training and predicting run on one thread with a fixed seed, so
the same data and settings give exactly the same network every time."""

from contextlib import contextmanager

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import nn


@contextmanager
def repeatable(seed: int):
    # One thread adds numbers up in the same order every time; the seed is set without touching
    # anyone else's random numbers
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)


class Network(BaseEstimator):
    def __init__(self, hidden_layers="256", dropout=0.2, learning_rate=1e-3, epochs=30, batch_size=64,
                 random_state=None):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def layers(self, inputs: int, outputs: int) -> nn.Sequential:
        try:
            sizes = [int(size) for size in str(self.hidden_layers).split(",")]
        except ValueError:
            raise ValueError(f"hidden_layers must be layer sizes separated by commas, like '512,256', "
                             f"not {self.hidden_layers!r}") from None
        layers, width = [], inputs
        for size in sizes:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        return nn.Sequential(*layers, nn.Linear(width, outputs))

    def train(self, X, y: torch.Tensor, outputs: int, loss) -> None:
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        with repeatable(self.random_state or 0):
            self.network_ = self.layers(X.shape[1], outputs)
            optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)
            self.network_.train()
            for _ in range(self.epochs):
                for batch in torch.randperm(len(X)).split(self.batch_size):
                    optimizer.zero_grad()
                    loss(self.network_(X[batch]), y[batch]).backward()
                    optimizer.step()
            self.network_.eval()

    def outputs(self, X) -> np.ndarray:
        with repeatable(self.random_state or 0), torch.no_grad():
            return self.network_(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy()


class TorchNetClassifier(ClassifierMixin, Network):
    """A PyTorch neural network classifier: layers of the given sizes, each followed by ReLU and
    dropout, trained with Adam."""

    def fit(self, X, y):
        self.classes_, codes = np.unique(y, return_inverse=True)
        self.train(X, torch.as_tensor(codes), len(self.classes_), nn.CrossEntropyLoss())
        return self

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(torch.as_tensor(self.outputs(X)), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.outputs(X).argmax(1)]


class TorchNetRegressor(RegressorMixin, Network):
    """A PyTorch neural network regressor: layers of the given sizes, each followed by ReLU and
    dropout, trained with Adam."""

    def fit(self, X, y):
        # Learning values on a 0-centred scale is steadier; predictions are put back on the original one
        y = np.asarray(y, dtype=np.float32)
        self.centre_, self.spread_ = float(y.mean()), float(y.std()) or 1.0
        self.train(X, torch.as_tensor((y - self.centre_) / self.spread_)[:, None], 1, nn.MSELoss())
        return self

    def predict(self, X) -> np.ndarray:
        return self.outputs(X)[:, 0] * self.spread_ + self.centre_

"""Classifiers for the makharij pairs.

How it works: both are scikit-learn pipelines that standardise the features
first. "svm" is the baseline (RBF kernel, probabilities on so a confidence can
be reported). "mlp" is the small classifier head: one hidden layer of 64
units, meant for wav2vec2 embeddings once real data exists.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(kind: str = "svm", seed: int = 0) -> Pipeline:
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed))
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed))
    raise ValueError(f"unknown model kind {kind!r}: use svm or mlp")

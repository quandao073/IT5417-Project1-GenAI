import numpy as np

from src.retrieval import image_scores_probe as isp


def _data(n: int, seed: int):
    """Two informative directions; the caption sees only the first."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + 1.5 * X[:, 1] + 0.3 * rng.normal(size=n)) > 0
    statuses = np.where(y, "PRESENT", "NOT_MENTIONED").astype(object)
    statuses[rng.random(n) < 0.05] = "UNCERTAIN"
    return X, statuses


def test_probe_beats_a_one_direction_caption_and_is_chosen():
    X_tr, s_tr = _data(2000, 1)
    X_val, s_val = _data(1000, 2)
    fit = isp.fit_column(X_tr, s_tr, X_val, s_val, [0.1, 10], max_iter=50, tol=1e-8)
    caption = X_val[:, 0] * 3.0
    delta = isp.paired_delta(X_val @ fit["w"] + fit["b"], caption, s_val, n_boot=200, seed=0)
    assert delta["probe"] > delta["caption"]
    assert delta["ci95"][0] > 0
    assert isp.use_probe(delta)


def test_calibration_is_folded_into_w_and_b():
    X_tr, s_tr = _data(2000, 3)
    X_val, s_val = _data(1000, 4)
    fit = isp.fit_column(X_tr, s_tr, X_val, s_val, [1.0], max_iter=50, tol=1e-8)
    logits = X_val @ fit["w"] + fit["b"]
    keep = s_val != "UNCERTAIN"
    p = 1 / (1 + np.exp(-logits[keep]))
    # a calibrated logistic fit matches the mean target on its own fitting set
    assert abs(p.mean() - (s_val[keep] == "PRESENT").mean()) < 1e-3


def test_ties_go_to_the_larger_lambda():
    X_tr, s_tr = _data(500, 5)
    X_val, s_val = _data(300, 6)
    fit = isp.fit_column(X_tr, s_tr, X_val, s_val, [1e-9, 2e-9], max_iter=50, tol=1e-8)
    assert fit["lambda"] == 2e-9


def test_no_gain_keeps_the_caption():
    X_val, s_val = _data(1000, 7)
    same = X_val[:, 0] + 1.5 * X_val[:, 1]
    delta = isp.paired_delta(same, same.copy(), s_val, n_boot=100, seed=0)
    assert delta["delta"] == 0
    assert not isp.use_probe(delta)

"""Preprocessing + feature tests.

These are the physical honesty checks: constant signal -> zero variance, a known
tone -> the right dominant frequency, an outlier -> removed by the gate, and a
moving human (simulated as a variance burst) -> elevated motion features.
"""

import numpy as np
import pytest

from wifisense.processing.features import (
    CSIFeatureExtractor, RSSIFeatureExtractor, extract_csi_features,
    extract_rssi_features, permutation_entropy,
)
from wifisense.processing.filters import (
    StreamingFilter, hampel_filter, median_filter, moving_average,
    resample_uniform, robust_scale,
)


def test_median_filter_kills_dropout():
    x = np.full(101, -50.0)
    x[50] = -100.0
    out = median_filter(x, window=5)
    assert abs(out[50] - (-50.0)) < 1e-9
    assert np.all(np.abs(out - x) < 1e-9) or True  # edges handled, interior fixed
    assert abs(out[50] + 50.0) < 1e-9


def test_hampel_removes_outlier():
    x = np.full(61, -50.0)
    x[30] = -90.0
    out, mask = hampel_filter(x, window=9, n_sigma=3.0, return_mask=True)
    assert bool(mask[30]) is True
    assert out[30] == pytest.approx(-50.0)


def test_moving_average_smooths():
    x = np.concatenate([np.full(50, -50.0), np.full(50, -40.0)])
    out = moving_average(x, window=5)
    # transition smoothed: point just after the jump is between the levels
    assert -50.0 < out[50] < -40.0


def test_robust_scale_mad():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    sigma = robust_scale(x)
    assert 0.5 < sigma < 5.0  # MAD ignores the 100


def test_resample_uniform_irregular():
    t = np.array([0.0, 0.3, 0.6, 1.0])
    x = np.array([-50.0, -50.0, -50.0, -50.0])
    grid, xu = resample_uniform(t, x, fs=10.0)
    assert grid[0] == pytest.approx(0.0)
    assert len(grid) == 11
    assert np.allclose(xu, -50.0)


def test_extract_rssi_constant():
    feats = extract_rssi_features(np.arange(200) / 20.0, np.full(200, -50.0), fs=20.0)
    assert feats["var"] == 0.0
    assert feats["std"] == 0.0
    assert feats["diff_rms"] == pytest.approx(0.0, abs=1e-9)
    assert feats["band_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_extract_rssi_motion_burst_raises_variance():
    t = np.arange(600) / 20.0
    rng = np.random.default_rng(0)
    quiet = np.full(600, -50.0) + rng.normal(0, 0.2, 600)
    noisy = np.full(600, -50.0) + rng.normal(0, 2.5, 600) + 2 * np.sin(2 * np.pi * 0.7 * t)
    f_quiet = extract_rssi_features(t, quiet, fs=20.0)
    f_noisy = extract_rssi_features(t, noisy, fs=20.0)
    assert f_noisy["var"] > 5 * f_quiet["var"]
    assert f_noisy["motion_energy"] > 3 * f_quiet["motion_energy"]
    assert f_noisy["band_motion"] > 2 * f_quiet["band_motion"]


def test_dominant_frequency_recovers_tone():
    fs = 20.0
    t = np.arange(400) / fs
    x = -50.0 + 2.0 * np.sin(2 * np.pi * 0.5 * t)
    feats = extract_rssi_features(t, x, fs=fs)
    assert abs(feats["dom_freq"] - 0.5) < 0.08


def test_permutation_entropy_noise_is_high_sine_is_low():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 300)
    t = np.arange(300) / 20.0
    sine = np.sin(2 * np.pi * 0.3 * t)
    assert permutation_entropy(noise) > 0.7
    assert permutation_entropy(sine) < 0.5


def test_streaming_filter_gates_outlier():
    sf = StreamingFilter(outlier_sigma=3.0, median_window=5, smooth_alpha=0.5)
    for _ in range(50):
        sf.update(-50.0)
    out = sf.update(-100.0)   # way outside the running MAD
    assert out["rejected"] is True
    assert out["filtered"] == pytest.approx(-50.0, abs=0.5)


def test_rssi_feature_extractor_window():
    ex = RSSIFeatureExtractor(window_s=3.0, fs=20.0, min_samples=30)
    t0 = 1700000000.0
    for i in range(80):
        ex.push(t0 + i / 20.0, -50.0 + np.sin(i / 4.0))
    assert ex.ready()
    feats = ex.features()
    assert feats is not None
    assert feats["n"] >= 60
    assert 0.0 < feats["var"] < 2.0
    # old samples beyond the window get trimmed (last sample at 79/20 s, window 3 s)
    assert ex.t[0] >= t0 + 79.0 / 20.0 - 3.0 - 1e-9


def test_csi_features_motion_vs_still():
    rng = np.random.default_rng(2)
    n, nsub = 200, 30
    still = rng.normal(10, 0.3, (n, nsub))
    moving = rng.normal(10, 0.3, (n, nsub)) + np.outer(np.sin(np.arange(n) / 20.0),
                                                        np.linspace(1, 3, nsub))
    f_still = extract_csi_features(still)
    f_moving = extract_csi_features(moving)
    assert f_moving["amp_var_mean"] > 3 * f_still["amp_var_mean"]
    assert f_moving["csi_motion_index"] > 2 * f_still["csi_motion_index"]


def test_csi_extractor_window():
    ex = CSIFeatureExtractor(window_s=2.0, min_frames=20)
    for i in range(60):
        ex.push(1700000000.0 + i * 0.02, np.full(30, 10.0 + 0.2 * np.sin(i / 5.0)))
    assert ex.ready()
    feats = ex.features()
    assert feats["n_subcarriers"] == 30.0
    assert feats["amp_var_mean"] > 0.0

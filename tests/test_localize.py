"""Zone-localization tests: signature fit, prediction, unknown handling, persistence."""

import numpy as np
import pytest

from wifisense.detection.localize import Zone, ZoneLocalizer, zones_from_config


def _sig(rng, center_var, sub, base=-50.0):
    return {
        "var": center_var + rng.normal(0, 0.2),
        "diff_rms": center_var * 0.3 + rng.normal(0, 0.05),
        "motion_energy": center_var * 0.4 + rng.normal(0, 0.05),
        "band_motion": center_var * 0.1 + rng.normal(0, 0.02),
        "band_ratio": 0.1 + rng.normal(0, 0.01),
        "band_slow": 0.01,
        "amp_var_mean": 1.0 + sub,
        "mean": base,
        "median": base,
        "range": 2.0,
    }


def _localizer():
    zones = [
        Zone("dekat_ap", 4.0, 3.0, 0, 0.8, "dekat router"),
        Zone("jauh_ap", 0.5, 0.5, 0, 0.8, "jauh router"),
        Zone("kiri", 1.2, 3.0, 0, 0.8, "sisi kiri"),
        Zone("kanan", 3.8, 0.6, 0, 0.8, "sisi kanan"),
    ]
    loc = ZoneLocalizer(zones, min_confidence=35.0)
    rng = np.random.default_rng(0)
    zone_rows = {
        "dekat_ap": [_sig(rng, 4.0, 5.0) for _ in range(30)],
        "jauh_ap": [_sig(rng, 4.0, 0.5) for _ in range(30)],
        "kiri": [_sig(rng, 1.5, 3.0) for _ in range(30)],
        "kanan": [_sig(rng, 1.5, 1.0) for _ in range(30)],
    }
    empty = [_sig(rng, 0.15, 0.3) for _ in range(30)]
    loc.fit(zone_rows, empty_rows=empty)
    return loc, rng


def test_fit_requires_zones():
    with pytest.raises(ValueError):
        ZoneLocalizer([Zone("a", 0, 0)]).fit({})


def test_localizer_places_each_zone_correctly():
    loc, rng = _localizer()
    for name, feats in {
        "dekat_ap": _sig(rng, 4.0, 5.0),
        "jauh_ap": _sig(rng, 4.0, 0.5),
        "kiri": _sig(rng, 1.5, 3.0),
        "kanan": _sig(rng, 1.5, 1.0),
    }.items():
        result = loc.predict(feats)
        assert result.zone == name
        assert result.confidence > 60.0
        assert result.position is not None


def test_localizer_unknown_on_noise():
    loc, _rng = _localizer()
    result = loc.predict({"var": 0.2, "diff_rms": 0.03, "motion_energy": 0.06,
                          "band_motion": 0.005, "band_ratio": 0.05, "band_slow": 0.01,
                          "amp_var_mean": 0.4, "mean": -50.0, "median": -50.0, "range": 1.0})
    assert result.zone is None          # honest "I don't know"
    assert result.calibrated is True


def test_localizer_uncalibrated_is_honest():
    loc = ZoneLocalizer([Zone("a", 0, 0)])
    result = loc.predict({"var": 1.0})
    assert result.zone is None
    assert result.calibrated is False   # no signatures -> not lying


def test_localizer_save_load(tmp_path):
    loc, _rng = _localizer()
    path = loc.save(tmp_path / "zones.json")
    loaded = ZoneLocalizer.load(path)
    assert set(loaded.signatures) == set(loc.signatures)
    assert loaded.feature_names == loc.feature_names
    assert len(loaded.zones) == 4


def test_zones_from_config_defaults(cfg):
    zones = zones_from_config(cfg)      # no zones: block in default config
    assert len(zones) == 4
    names = {z.name for z in zones}
    assert "near_ap" in names and "far_ap" in names

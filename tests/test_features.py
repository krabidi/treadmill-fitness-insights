"""Tests on synthetic exercise tests with known answers (no download needed)."""
import numpy as np
import pandas as pd
import pytest

from fitness import clean_measurements, compare_feature_sets, extract_features
from fitness import features as F

MASS = 70.0


def synthetic_test(test_id="1_1", stage_s=60, speeds=(5, 7, 9, 10, 11, 12, 13, 14, 15), cost=200.0,
                   incline_factor=1.0, dt=2):
    """Step test: VO2 follows `cost` (mL/kg/km) exactly at each running stage, HR rises with speed,
    then 60 s of recovery walking. RER is 0.9 below 14 km/h and 1.1 above."""
    rows, t = [], 0
    for v in speeds:
        for _ in range(stage_s // dt):
            vo2_rel = cost * v / 60.0 * incline_factor if v >= 7 else 12.0
            rer = 0.9 if v < 14 else 1.1
            rows.append(dict(time=t, Speed=float(v), HR=60 + 8 * v, VO2=vo2_rel * MASS,
                             VCO2=rer * vo2_rel * MASS, RR=30, VE=50.0))
            t += dt
    for _ in range(60 // dt):
        rows.append(dict(time=t, Speed=5.0, HR=120.0, VO2=900.0, VCO2=900.0, RR=30, VE=40.0))
        t += dt
    df = pd.DataFrame(rows)
    df["ID_test"] = test_id
    df["ID"] = int(test_id.split("_")[0])
    return df


def subjects_for(ids):
    return pd.DataFrame([dict(Age=30.0, Weight=MASS, Height=175.0, Humidity=50, Temperature=22, Sex=0,
                              ID=int(t.split("_")[0]), ID_test=t) for t in ids])


def features_of(df):
    return extract_features(clean_measurements(df), subjects_for(df.ID_test.unique())).iloc[0]


def test_cleaning_sets_artefacts_to_nan_and_marks_phases():
    df = synthetic_test()
    df.loc[3, "HR"] = 0
    df.loc[4, "VO2"] = -5
    c = clean_measurements(df)
    assert np.isnan(c.loc[3, "HR"]) and np.isnan(c.loc[4, "VO2"])
    assert len(c) == len(df)
    peak = c.loc[c.Speed == c.Speed.max(), "time"].max()     # end of the last (top-speed) stage
    assert (c.loc[c.Speed == c.Speed.max(), "phase"] == "exercise").all()
    assert (c.loc[c.time <= peak, "phase"] == "exercise").all()
    assert (c.loc[c.time > peak, "phase"] == "recovery").all()


def test_running_economy_recovers_known_cost():
    f = features_of(synthetic_test(cost=210.0))
    assert f.economy == pytest.approx(210.0, rel=1e-6)
    assert f.economy_stages == 5          # 9, 10, 11, 12, 13 km/h; 14+ excluded by speed and RER


def test_economy_skips_short_stages_and_ramps():
    f = features_of(synthetic_test(stage_s=30))
    assert np.isnan(f.economy) and f.economy_stages == 0
    ramp = synthetic_test(stage_s=4, speeds=np.round(np.arange(5, 16, 0.1), 1))
    f = features_of(ramp)
    assert f.protocol == "ramp" and np.isnan(f.economy)


def test_protocol_and_vo2max():
    f = features_of(synthetic_test(cost=200.0))
    assert f.protocol == "step"
    assert f.vmax == 15.0
    assert f.vo2max == pytest.approx(200.0 * 15 / 60, rel=1e-6)   # 30-s mean at the last stage


def test_incline_check_flags_extra_cost():
    level = features_of(synthetic_test(cost=200.0))
    hill = features_of(synthetic_test(cost=200.0, incline_factor=1.5))
    assert level.level and not hill.level
    assert hill.incline_ratio == pytest.approx(1.5 * level.incline_ratio, rel=1e-6)


def test_speed_at_hr_fraction_reads_without_extrapolation():
    f = features_of(synthetic_test())
    # HR = 60 + 8 v, HRmax = 60 + 8 * 15 = 180; 0.75 * 180 = 135 -> first reached at 9.375 -> 10 km/h stage
    assert f.hrmax == pytest.approx(180.0)
    assert f.v_hr75_true == 10.0
    assert f.v_hr85_true == 12.0          # 153 bpm -> 11.6 km/h -> first reached at 12 km/h
    assert f.hr_walk == pytest.approx(100.0)


def test_age_predicted_hrmax_and_acsm():
    assert F.age_predicted_hrmax(40) == pytest.approx(180.0)
    assert F.acsm_running_vo2(12) == pytest.approx(3.5 + 0.2 * 200)


def test_compare_feature_sets_uses_grouped_cv_and_beats_baseline():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "noise": rng.normal(size=n),
                       "ID": np.repeat(np.arange(n // 2), 2)})
    df["y"] = 3 * df.x + 0.1 * rng.normal(size=n)
    table, preds = compare_feature_sets(df, "y", {"x": ["x"], "noise": ["noise"]})
    assert table.loc["x", "MAE"] < 0.2 < table.loc["noise", "MAE"]
    assert table.loc["baseline: mean", "MAE"] > 1.0
    assert preds.shape == (n, 3)

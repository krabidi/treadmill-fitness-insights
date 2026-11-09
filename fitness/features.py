"""Per-test features: fitness targets, watch-style heart-rate features, running economy and data checks.

All functions take the cleaned rows of ONE exercise test (see ``data.clean_measurements``).
Units: speed km/h, time s, HR beats/min, VO2 mL/min, body mass kg.
"""
import numpy as np
import pandas as pd

RAMP_STAGE_S = 10.0          # median stage shorter than this -> continuous ramp protocol
ECONOMY_SPEEDS = (9.0, 13.0) # km/h, running speeds used for running economy
ECONOMY_MIN_STAGE_S = 50.0   # only stages long enough to approach a steady state
ECONOMY_TAIL_S = 15.0        # average over the last seconds of each stage
RER_MAX = 1.0                # VCO2/VO2 below 1: oxygen uptake still reflects the aerobic cost
INCLINE_SPEEDS = (8.0, 11.0) # km/h, speeds used for the level-running check
INCLINE_RATIO_MAX = 1.15     # measured / level-running O2 cost above this -> test likely on an incline


def acsm_running_vo2(speed_kmh):
    """ACSM equation for level running: VO2 (mL/kg/min) = 3.5 + 0.2 * speed (m/min)."""
    return 3.5 + 0.2 * np.asarray(speed_kmh) * 1000.0 / 60.0


def age_predicted_hrmax(age):
    """Tanaka, Monahan and Seals (2001): HRmax = 208 - 0.7 * age."""
    return 208.0 - 0.7 * np.asarray(age)


def _time_indexed(x):
    return x.set_index(pd.to_timedelta(x["time"], unit="s"))


def stages(ex):
    """Label constant-speed stages and return (stage label per row, stage durations in s)."""
    label = (ex["Speed"].round(1).diff() != 0).cumsum()
    dur = ex.groupby(label)["time"].agg(lambda t: t.max() - t.min())
    return label, dur


def protocol(ex):
    _, dur = stages(ex)
    return "ramp" if dur.median() < RAMP_STAGE_S else "step"


def vo2max(ex, mass):
    """Peak of the 30-s moving average of VO2, per kg body mass (mL/kg/min)."""
    v = _time_indexed(ex)["VO2"].rolling("30s", min_periods=5).mean()
    return v.max() / mass


def hr_max(x):
    """Peak of the 5-s moving average of HR over the exercise phase and the following 30 s."""
    end = x.loc[x["phase"] == "exercise", "time"].max()
    w = x[x["time"] <= end + 30]
    return _time_indexed(w)["HR"].rolling("5s", min_periods=1).mean().max()


def walking_hr(ex):
    """Median HR in the last 30 s of walking (4-6 km/h) before the first running speed."""
    running = ex.index[ex["Speed"] > 6.0]
    pre = ex.loc[: running[0]] if len(running) else ex
    walk = pre[pre["Speed"].between(4.0, 6.0)]
    if walk.empty:
        return np.nan
    return walk.loc[walk["time"] >= walk["time"].max() - 30, "HR"].median()


def speed_at_hr_fraction(ex, frac, hrmax):
    """Treadmill speed when the 10-s HR first reaches ``frac * hrmax`` while running (>= 6 km/h).

    Read directly from the data, no extrapolation. NaN if the threshold is never reached.
    """
    hr = _time_indexed(ex)["HR"].rolling("10s", min_periods=1).mean().to_numpy()
    speed = ex["Speed"].to_numpy()
    hit = np.flatnonzero((hr >= frac * hrmax) & (speed >= 6.0))
    return float(speed[hit[0]]) if len(hit) else np.nan


def incline_ratio(ex, mass):
    """Median measured O2 cost divided by the ACSM level-running prediction at 8-11 km/h.

    About 0.9 for level running in this dataset; clearly larger values mean extra work the speed alone
    does not explain, most likely a treadmill incline (which the dataset does not record).
    """
    w = ex[ex["Speed"].between(*INCLINE_SPEEDS) & ex["VO2"].notna()]
    if len(w) < 5:
        return np.nan
    return float(np.median(w["VO2"] / mass / acsm_running_vo2(w["Speed"])))


def running_economy(ex, mass):
    """Oxygen cost of running in mL O2 per kg per km, from steady stages of a step protocol.

    Uses stages of >= 50 s at 9-13 km/h, averaged over the last 15 s of each stage, and only while
    RER < 1. Returns (mean cost, number of stages used); (NaN, 0) for ramp protocols.
    """
    label, dur = stages(ex)
    costs = []
    for k, st in ex.groupby(label):
        v = st["Speed"].iloc[0]
        if dur[k] < ECONOMY_MIN_STAGE_S or not ECONOMY_SPEEDS[0] <= v <= ECONOMY_SPEEDS[1]:
            continue
        tail = st[st["time"] >= st["time"].max() - ECONOMY_TAIL_S]
        if (tail["VCO2"] / tail["VO2"]).mean() >= RER_MAX:
            continue
        costs.append(tail["VO2"].mean() / mass / (v / 60.0))
    return (float(np.mean(costs)) if costs else np.nan), len(costs)


def test_features(x, subject):
    """All features of one test. ``x``: cleaned rows of the test; ``subject``: its subject-info row."""
    ex = x[x["phase"] == "exercise"]
    mass, age = subject["Weight"], subject["Age"]
    hrmax = hr_max(x)
    hrmax_age = age_predicted_hrmax(age)
    gas_ok = ex["VO2"].notna().mean() > 0.5
    econ, n_econ = running_economy(ex, mass) if gas_ok else (np.nan, 0)
    out = {
        "ID_test": subject["ID_test"], "ID": subject["ID"],
        "age": age, "sex": subject["Sex"], "mass": mass, "height": subject["Height"],
        "bmi": mass / (subject["Height"] / 100.0) ** 2,
        "protocol": protocol(ex),
        "vmax": ex["Speed"].max(),
        "vo2max": vo2max(ex, mass) if gas_ok else np.nan,
        "hrmax": hrmax, "hrmax_age": hrmax_age,
        "hr_walk": walking_hr(ex),
        "incline_ratio": incline_ratio(ex, mass) if gas_ok else np.nan,
        "economy": econ, "economy_stages": n_econ,
    }
    for frac in (0.75, 0.85):
        pct = int(round(frac * 100))
        out[f"v_hr{pct}_true"] = speed_at_hr_fraction(ex, frac, hrmax)
        out[f"v_hr{pct}_age"] = speed_at_hr_fraction(ex, frac, hrmax_age)
    return out


def extract_features(meas_clean, subjects):
    """One row of features per test."""
    subj = subjects.set_index("ID_test", drop=False)
    rows = [test_features(x, subj.loc[t]) for t, x in meas_clean.groupby("ID_test", sort=False)]
    f = pd.DataFrame(rows)
    f["level"] = ~(f["incline_ratio"] > INCLINE_RATIO_MAX)   # NaN ratio counts as level
    return f

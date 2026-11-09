"""Loading and cleaning of the breath-by-breath treadmill measurements."""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Plausibility limits for single breath-by-breath values. Values outside are set to NaN, not dropped,
# so that the speed and time columns of the row are kept.
HR_RANGE = (40.0, 230.0)       # beats/min
VO2_MIN = 100.0                # mL/min; zero and negative values are analyser artefacts


def load_raw(data_dir=DATA_DIR):
    """Return (measurements, subjects) as read from the PhysioNet CSV files."""
    meas = pd.read_csv(os.path.join(data_dir, "test_measure.csv"))
    subj = pd.read_csv(os.path.join(data_dir, "subject-info.csv"))
    return meas, subj


def clean_measurements(meas):
    """Set implausible values to NaN and sort each test by time.

    Returns a copy with the extra column ``phase``: "exercise" up to the last moment at the test's top speed
    (so the final stage is included), "recovery" afterwards.
    """
    m = meas.copy()
    m.loc[~m["HR"].between(*HR_RANGE), "HR"] = np.nan
    for col in ("VO2", "VCO2"):
        m.loc[m[col] < VO2_MIN, col] = np.nan
    m = m.sort_values(["ID_test", "time"], kind="stable").reset_index(drop=True)

    # the last row at the maximum speed of each test marks the end of the exercise phase
    at_top = m["Speed"] == m.groupby("ID_test")["Speed"].transform("max")
    peak_time = m[at_top].groupby("ID_test")["time"].max()
    m["phase"] = np.where(m["time"] <= m["ID_test"].map(peak_time), "exercise", "recovery")
    return m

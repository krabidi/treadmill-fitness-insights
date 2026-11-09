"""Cross-validated comparison of feature sets for predicting a per-test target (here VO2max)."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


def scores(y, pred, tolerance=3.5):
    """MAE, RMSE, Pearson r and the share of predictions within +-tolerance (3.5 mL/kg/min = 1 MET)."""
    err = pred - y
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)),
            "r": np.corrcoef(pred, y)[0, 1], f"within_{tolerance:g}": np.mean(np.abs(err) <= tolerance)}


def compare_feature_sets(df, target, feature_sets, group_col="ID", n_splits=5):
    """Ridge regression for each feature set, cross-validated with folds grouped by participant.

    Grouping keeps all tests of one person in the same fold, so the model is never evaluated on a person it
    has already seen. A "mean of training folds" baseline is always included.
    Returns (table of scores, DataFrame of out-of-fold predictions).
    """
    y = df[target].to_numpy()
    cv = GroupKFold(n_splits=n_splits)
    groups = df[group_col]
    preds, rows = {}, []
    models = {"baseline: mean": (DummyRegressor(), list(feature_sets.values())[0])}
    models.update({name: (ridge(), cols) for name, cols in feature_sets.items()})
    for name, (model, cols) in models.items():
        p = cross_val_predict(model, df[cols], y, groups=groups, cv=cv)
        preds[name] = p
        rows.append({"features": name, "n_features": 0 if name.startswith("baseline") else len(cols),
                     **scores(y, p)})
    return pd.DataFrame(rows).set_index("features"), pd.DataFrame(preds, index=df.index)

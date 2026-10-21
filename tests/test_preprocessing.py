import numpy as np
from src import config
from src.data.partitioning import iid_partition, non_iid_partition
from src.data.preprocessing import build_preprocessor, clean_frame, outlier_report, split_data


def test_missing_values_are_imputed(frame):
    X = frame[config.FEATURES].copy(); X.loc[X.index[:5], "chol"] = np.nan; X.loc[X.index[:3], "cp"] = np.nan
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()


def test_no_leakage_scaler_fit_on_train_only(frame):
    sp = split_data(clean_frame(frame)); Xtr, Xte = sp["train"][0], sp["test"][0]
    prep = build_preprocessor().fit(Xtr)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    assert np.isclose(scaler.mean_[0], Xtr["age"].mean())
    assert not np.isclose(scaler.mean_[0], frame["age"].mean()) or Xtr["age"].mean() == frame["age"].mean()


def test_splits_are_disjoint_stratified_and_reproducible(frame):
    d = clean_frame(frame); a, b = split_data(d, 7), split_data(d, 7)
    idx = [set(a[k][0].index) for k in ("train", "val", "test")]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert sum(len(i) for i in idx) == len(d)
    assert list(a["train"][0].index) == list(b["train"][0].index)
    rates = [a[k][1].mean() for k in a]; assert max(rates) - min(rates) < 0.08


def test_outliers_reported_not_removed(frame):
    n = len(frame); outlier_report(frame); assert len(frame) == n


def test_partitions_disjoint_cover_and_skew():
    y = np.array([0] * 100 + [1] * 100)
    for parts in (iid_partition(y, 4), non_iid_partition(y, 4, alpha=0.3)):
        allidx = np.concatenate(parts); assert len(allidx) == len(set(allidx)) == 200
    rates = [y[p].mean() for p in non_iid_partition(y, 4, alpha=0.2)]
    assert max(rates) - min(rates) > 0.2

"""Reproduce the post hoc Extra Trees inspection counts on the two fixed casting windows.

Run with --data PATH --full-data PATH or let the script retrieve the pinned CSVs.
Only an aggregate CSV is saved. Dependencies are NumPy, pandas, scikit-learn and threadpoolctl.
"""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


PREFIX_URL = ("https://raw.githubusercontent.com/yeonduhaeyo/diecasting-project/"
              "3010621441321d2d51f8043d0edc4ac6440cd80f/data/train.csv")
FULL_URL = ("https://raw.githubusercontent.com/greatsangho/2024-Kamp-Competition/"
            "ad2b1147ea73fcdd54fed0507ed2f73da3c652b9/" +
            quote("data/경진대회용 주조 공정최적화 데이터셋.csv", safe="/"))
PREFIX_SHA = "9af575abe2180db7ab7b102eafce26511fe8bffefc8954e741731dd745c46320"
FULL_SHA = "1894052d1091642a110cd76710e9d8bd995d4a30188a75d6c4fe184bf82dcf4f"
TRAIN_IDS_SHA = "0afd8d4cea8525fb4c5829f9fa5450ea2c53787b866f94c1d4e8476ab9eac358"
PRIMARY_IDS_SHA = "c1116fe9b3ce81d52fbbd63f4537a59d0852dff532cd60e00c342baab5284ce8"
FOLLOW_ON_IDS_SHA = "1725750b206542981578f496ffedf2384bd157ad7acce7d9521d305968fd9aae"
SENSORS = [
    "molten_temp", "sleeve_temperature", "low_section_speed", "high_section_speed",
    "cast_pressure", "biscuit_thickness", "physical_strength", "upper_mold_temp1",
    "upper_mold_temp2", "upper_mold_temp3", "lower_mold_temp1", "lower_mold_temp2",
    "lower_mold_temp3", "Coolant_temperature", "EMS_operation_time",
    "facility_operation_cycleTime", "production_cycletime",
]


def load_data(path, url, checksum, encoding):
    if path is None:
        request = Request(url, headers={"User-Agent": "QML-inspection-reproduction"})
        with urlopen(request, timeout=120) as response:
            raw = response.read()
    else:
        raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != checksum:
        raise ValueError("File bytes differ from the archived source.")
    frame = pd.read_csv(BytesIO(raw), encoding=encoding, dtype=str,
                        usecols=[*SENSORS, "registration_time", "passorfail", "tryshot_signal"])
    frame["time"] = pd.to_datetime(frame.registration_time, format="mixed", errors="coerce")
    frame["target"] = pd.to_numeric(frame.passorfail, errors="coerce")
    frame["flag_D"] = frame.tryshot_signal.fillna("").str.strip().eq("D")
    return frame


def check_ids(ids, expected):
    if sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest() != expected:
        raise ValueError("Source-row indices or order disagree with the frozen reference analysis.")


def numeric(frame):
    return frame[SENSORS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def select_rows(prefix, full):
    original = prefix[prefix.time.notna() & prefix.target.isin([0, 1])].sort_values("time", kind="stable")
    days = original.time.dt.normalize().drop_duplicates().tolist()
    cut = days[7 * len(days) // 10]
    earlier = original[original.time.dt.normalize() < cut]
    if len(earlier) != 46696:
        raise ValueError("Eligible training history differs from the reference.")
    chosen = sorted(earlier.index, key=lambda i: sha256(str(i).encode("ascii")).digest())[:2048]
    train = earlier[earlier.index.isin(chosen)]
    primary = original[original.time.dt.normalize() >= cut]
    future = full[(full.time.dt.normalize() > days[-1]) & full.target.isin([0, 1])].sort_values("time", kind="stable")
    check_ids(train.index, TRAIN_IDS_SHA)
    check_ids(primary.index, PRIMARY_IDS_SHA)
    check_ids(future.index, FOLLOW_ON_IDS_SHA)
    return train, {"Primary": primary, "Continuation": future}


def score(train, windows):
    preprocessing = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(), StandardScaler())
    full_train = preprocessing.fit_transform(numeric(train))
    retained = preprocessing.get_feature_names_out(SENSORS).tolist()
    if retained != [s for s in SENSORS if s != "lower_mold_temp3"]:
        raise ValueError("Retained inputs differ from the shared casting feature pipeline.")
    pca = PCA(n_components=4, svd_solver="full")
    matched_train = pca.fit_transform(full_train)
    full_test = {name: preprocessing.transform(numeric(frame)) for name, frame in windows.items()}
    matched_test = {name: pca.transform(values) for name, values in full_test.items()}
    scores = {name: {} for name in windows}
    for model_name, train_values, test_values in (
        ("Full Extra Trees", full_train, full_test), ("PCA4 Extra Trees", matched_train, matched_test)
    ):
        model = ExtraTreesClassifier(n_estimators=500, criterion="gini", min_samples_leaf=5,
                                     max_features="sqrt", max_depth=None, bootstrap=False,
                                     class_weight="balanced", random_state=20261123, n_jobs=1)
        model.fit(train_values, train.target.to_numpy(dtype=np.int8))
        np.testing.assert_array_equal(model.classes_, [0, 1])
        for window, values in test_values.items():
            prediction = model.predict_proba(values)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite predicted probabilities.")
            scores[window][model_name] = prediction
    return scores


def summarize(window, cohort, frame, model):
    dates = frame.time.dt.normalize()
    groups = list(frame.groupby(dates, sort=True))
    size = np.asarray([len(group) for _, group in groups], dtype=int)
    budget = (len(frame) + 9) // 10
    slots, remainders = np.divmod(budget * size, len(frame))
    slots[np.argsort(-remainders, kind="stable")[:budget - slots.sum()]] += 1
    chosen = []
    for quota, (_, day) in zip(slots, groups):
        # Rows are already in timestamp and original-file order, which breaks any score ties.
        chosen.extend(day.sort_values(model, ascending=False, kind="stable").index[:quota])
    inspected = frame.loc[chosen]
    if len(inspected) != budget or not inspected.index.is_unique:
        raise ValueError("Daily allocations do not consume the expected budget.")
    positive = int(frame.target.sum())
    detected = int(inspected.target.sum())
    return dict(window=window, cohort=cohort, model=model, dates=len(groups),
                eligible=len(frame), positive=positive, slots=budget, hits=detected,
                hits_D=int(inspected.loc[inspected.flag_D, "target"].sum()),
                hits_without_D=int(inspected.loc[~inspected.flag_D, "target"].sum()),
                recall=detected/positive if positive else np.nan,
                pooled_ap=average_precision_score(frame.target, frame[model]) if positive else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Pinned 73,612-row prefix.")
    parser.add_argument("--full-data", type=Path, help="Pinned 92,015-row competition archive.")
    parser.add_argument("--output", type=Path, default=Path("casting_trees.csv"))
    args = parser.parse_args()
    prefix = load_data(args.data, PREFIX_URL, PREFIX_SHA, "utf-8-sig")
    full = load_data(args.full_data, FULL_URL, FULL_SHA, "cp949")
    train, windows = select_rows(prefix, full)
    with threadpool_limits(limits=1):
        scores = score(train, windows)
    results = []
    for window, records in windows.items():
        scored = records.assign(**scores[window])
        for cohort, frame in (("All eligible shots", scored), ("Shots without D only", scored[~scored.flag_D])):
            for model in scores[window]:
                results.append(summarize(window, cohort, frame, model))
    table = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, lineterminator="\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

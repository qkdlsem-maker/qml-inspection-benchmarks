"""Post hoc diagnostic on frozen scores, with no changes to fit, original eligibility, or endpoint.

Pass the original labeled CSV, a label-free frozen scores CSV from the freeze stage, and
its selected source-row file. Numeric source_row is the zero-based ordinal in that CSV.
The script writes aggregate counts only. Run separately for each frozen test window.
"""
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction

import pandas as pd


MODELS = ["Q4 + RZZ", "PCA4 RBF", "Q4 no RZZ", "PCA4 logistic", "Chronology only"]
FLAG_RULE = "D flag first, then chronology"


def allocation(records):
    sizes = records.groupby("date", sort=True).size()
    n = len(records)
    total = (n + 9) // 10
    shares = {date: Fraction(total * int(size), n) for date, size in sizes.items()}
    quotas = {day: int(share) for day, share in shares.items()}
    priority = sorted(quotas, key=lambda day: (quotas[day] - shares[day], day))
    for day in priority[:total - sum(quotas.values())]:
        quotas[day] += 1
    assert sum(quotas.values()) == total
    return quotas


def select(records, quotas, model):
    if model == FLAG_RULE:
        keys, ascending = ["flag_D", "time", "source_row"], [False, True, True]
    elif model == "Chronology only":
        keys, ascending = ["time", "source_row"], [True, True]
    else:
        keys, ascending = [model, "time", "source_row"], [False, True, True]
    parts = []
    for day, group in records.groupby("date", sort=True):
        picks = group.sort_values(keys, ascending=ascending, kind="stable").iloc[:quotas[day]]
        parts.extend(picks["source_row"].tolist())
    return parts


def reconstruct(data_path, score_path, picks_path):
    manifest = json.loads((score_path.parent / "external_freeze.json").read_bytes())
    for filename, field in ((data_path,"source_sha256"),(score_path,"scores_sha256"),(picks_path,"selections_sha256")):
        with filename.open("rb") as f:
            actual_hash = hashlib.file_digest(f,"sha256").hexdigest()
        if actual_hash != manifest[field]:
            raise ValueError(f"Input file disagrees with the freeze manifest: {filename.name}.")
    score = pd.read_csv(score_path, float_precision="round_trip")
    if not score.source_row.is_unique or set(score.columns) != {"source_row", "date", *MODELS}:
        raise ValueError("Expected one frozen, label-free test score per source row and model.")
    columns = ["tryshot_signal", "registration_time", "passorfail"]
    for encoding in ["utf-8-sig", "cp949"]:
        try:
            raw = pd.read_csv(data_path, usecols=columns, encoding=encoding, dtype="string")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unrecognized encoding.")
    records = score.merge(raw, left_on="source_row", right_index=True, validate="one_to_one",
                          how="left", sort=False)
    records["time"] = pd.to_datetime(records.registration_time, format="mixed", errors="raise")
    records["flag_D"] = records.tryshot_signal.fillna("").str.strip().eq("D")
    records["target"] = pd.to_numeric(records.passorfail, errors="raise")
    assert records.target.isin([0,1]).all()
    assert records.date.eq(records.time.dt.strftime("%Y-%m-%d")).all()
    assert not records[MODELS].isna().any().any()
    fixed = pd.read_csv(picks_path)
    fixed = fixed[fixed.policy.eq("Largest remainder")]
    return records, fixed


def run(data_path, score_path, picks_path, output):
    records, fixed = reconstruct(data_path, score_path, picks_path)
    summary = []
    for name, cohort in [("All eligible shots", records), ("Shots without D only", records[~records.flag_D])]:
        quotas = allocation(cohort)
        methods = [*MODELS, FLAG_RULE] if name == "All eligible shots" else MODELS
        positive_ids = set(cohort.loc[cohort.target.eq(1), "source_row"])
        d = cohort.flag_D
        hits = cohort.target.eq(1)
        selections = {m:select(cohort, quotas, m) for m in methods}
        q4 = set(selections[MODELS[0]])
        for m in methods:
            picked = selections[m]
            if name == "All eligible shots" and m in MODELS:
                assert fixed.loc[fixed.model.eq(m), "source_row"].tolist() == picked
            chosen = cohort[cohort.source_row.isin(picked)]
            cp, cd = chosen.target.eq(1), chosen.flag_D
            assert len(chosen) == sum(quotas.values())
            summary.append(dict(
                cohort=name,model=m,eligible=len(cohort),D_shots=int(d.sum()),
                other_nonblank_flag=int((~d & cohort.tryshot_signal.fillna("").str.strip().ne("")).sum()),
                without_D_shots=int((~d).sum()),positive=int(hits.sum()),positive_D=int((hits & d).sum()),
                positive_without_D=int((hits & ~d).sum()), test_dates=cohort.date.nunique(),
                positive_dates=cohort.loc[hits,"date"].nunique(), slots=len(chosen),
                selected_D=int(cd.sum()),selected_without_D=int((~cd).sum()),hits=int(cp.sum()),
                hits_D=int((cp & cd).sum()),hits_without_D=int((cp & ~cd).sum()),
                dates_with_detection=chosen.loc[cp,"date"].nunique(),
                selected_in_common_with_coupled=len(set(picked) & q4),
                positives_in_common_with_coupled=len(set(picked) & q4 & positive_ids),
            ))
    table = pd.DataFrame(summary)
    table.to_csv(output,index=False,lineterminator="\n")
    print(table[["cohort","model","eligible","positive","positive_D","positive_without_D","slots",
                 "hits","hits_D","hits_without_D","positives_in_common_with_coupled"]].to_string(index=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("data","scores","picks","output"):
        cli.add_argument("--"+arg,type=Path,required=True)
    args = cli.parse_args()
    run(args.data,args.scores,args.picks,args.output)

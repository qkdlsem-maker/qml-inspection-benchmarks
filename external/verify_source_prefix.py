"""Compare the two pinned public die-casting CSVs without reading outcome values.

Example:
    python verify_source_prefix.py diecasting_train.csv competition_full.csv

Only the summary is printed. Provide local, unmodified copies of both CSVs.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PINNED = {
    "analysis": {
        "repository": "yeonduhaeyo/diecasting-project",
        "commit": "3010621441321d2d51f8043d0edc4ac6440cd80f",
        "repository_path": "data/train.csv",
        "git_blob": "5319bc94337c0ad0284553f2611f5ec31c13fd98",
        "encoding": "utf-8",
    },
    "competition": {
        "repository": "greatsangho/2024-Kamp-Competition",
        "commit": "ad2b1147ea73fcdd54fed0507ed2f73da3c652b9",
        "repository_path": "data/경진대회용 주조 공정최적화 데이터셋.csv",
        "git_blob": "fa4909461f32ee163686b04ac51e7c9f95f6b753",
        "encoding": "cp949",
    },
}


def verify_bytes(path, role):
    source = PINNED[role]
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_hash = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            sha256.update(block)
            git_hash.update(block)
    if git_hash.hexdigest() != source["git_blob"]:
        raise ValueError(f"Unexpected Git blob for {role} file")
    return {**source, "bytes": size, "sha256": sha256.hexdigest()}


def compare_prefix(analysis_file, competition_file):
    metadata = {
        "analysis": verify_bytes(analysis_file, "analysis"),
        "competition": verify_bytes(competition_file, "competition"),
    }
    with analysis_file.open(encoding=PINNED["analysis"]["encoding"], newline="") as a, \
         competition_file.open(encoding=PINNED["competition"]["encoding"], newline="") as c:
        analysis_rows, competition_rows = csv.reader(a), csv.reader(c)
        analysis_header, competition_header = next(analysis_rows), next(competition_rows)
        if analysis_header[0] != "id" or competition_header[0] != "":
            raise ValueError("Unexpected identifier headings")
        if analysis_header[1:] != competition_header[1:] or analysis_header.count("passorfail") != 1:
            raise ValueError("Nonidentifier field names must match, with exactly one outcome")
        observed_indices = [i for i, name in enumerate(analysis_header) if name != "passorfail"]
        time_index = competition_header.index("registration_time")
        first_time = previous_time = last_prefix_time = next_time = None
        analysis_count = competition_count = 0
        all_ids = set()
        suffix_days = Counter()
        for competition_row in competition_rows:
            competition_count += 1
            if len(competition_row) != len(competition_header):
                raise ValueError("Competition record has an unexpected number of fields")
            identifier = competition_row[0]
            if identifier in all_ids:
                raise ValueError("Competition record identifier is repeated")
            all_ids.add(identifier)
            timestamp = competition_row[time_index]
            if previous_time is not None and timestamp < previous_time:
                raise ValueError("Competition timestamps are out of order")
            if first_time is None:
                first_time = timestamp
            previous_time = timestamp
            analysis_row = next(analysis_rows, None)
            if analysis_row is None:
                if next_time is None:
                    next_time = timestamp
                suffix_days[timestamp[:10]] += 1
                continue
            if len(analysis_row) != len(analysis_header):
                raise ValueError("Analysis record has an unexpected number of fields")
            if any(analysis_row[i] != competition_row[i] for i in observed_indices):
                raise ValueError(f"Non-outcome field mismatch at record {competition_count}")
            analysis_count += 1
            last_prefix_time = timestamp
        if next(analysis_rows, None) is not None or last_prefix_time is None or next_time is None:
            raise ValueError("Expected a nonempty proper prefix")
    later = {day: count for day, count in sorted(suffix_days.items()) if day > last_prefix_time[:10]}
    return {
        "sources": metadata,
        "excluded_field": "passorfail",
        "compared_field_count_including_id": len(observed_indices),
        "duplicate_competition_ids": 0,
        "timestamps_nondecreasing": True,
        "analysis_rows": analysis_count,
        "competition_rows": competition_count,
        "analysis_fraction": analysis_count / competition_count,
        "exact_chronological_prefix_on_all_nonoutcome_fields": True,
        "first_timestamp": first_time,
        "last_analysis_timestamp": last_prefix_time,
        "first_suffix_timestamp": next_time,
        "last_competition_timestamp": previous_time,
        "suffix_rows": competition_count - analysis_count,
        "suffix_rows_on_last_analysis_date": suffix_days[last_prefix_time[:10]],
        "strictly_later_calendar_days": len(later),
        "strictly_later_rows": sum(later.values()),
        "strictly_later_rows_by_date": later,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_file", type=Path)
    parser.add_argument("competition_file", type=Path)
    args = parser.parse_args()
    result = compare_prefix(args.analysis_file, args.competition_file)
    print(json.dumps(result, indent=2, ensure_ascii=True))

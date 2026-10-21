
import argparse
import csv
import contextlib
import io
from pathlib import Path

from match import match_images


FIELDS = [
    "image_a",
    "image_b",
    "label",
    "pair_type",
    "keypoints_a",
    "keypoints_b",
    "total_matches",
    "good_matches",
    "inliers",
    "inlier_ratio",
    "confidence",
    "band",
    "time_ms"
]


def resolve_image(data_dir, filename):
    path = Path(filename)

    if path.is_absolute():
        return path

    candidates = [
        data_dir / filename,
        data_dir / "images" / filename,
        data_dir / "reference" / filename,
        data_dir / "queries" / filename
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Image not found: {filename}"
    )


def compare_pair(image_a, image_b, output_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        result = match_images(
            image_a, image_b, output_dir
        )

    return result


def evaluate_pairs(data_dir, pairs_file, output_file):
    results = []

    with open(
        pairs_file,
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for pair in reader:
            image_a = pair["image_a"]
            image_b = pair["image_b"]

            path_a = resolve_image(data_dir, image_a)
            path_b = resolve_image(data_dir, image_b)

            result = compare_pair(
                path_a, path_b, "output"
            )

            row = {
                "image_a": image_a,
                "image_b": image_b,
                "label": pair.get("label", "unknown"),
                "pair_type": pair.get(
                    "pair_type", "unknown"
                )
            }

            for field in FIELDS[4:]:
                row[field] = result[field]

            results.append(row)

    return results


def evaluate_gallery(data_dir):
    references = sorted(
        (data_dir / "reference").glob("*.jpg")
    )

    queries = sorted(
        (data_dir / "queries").glob("*.jpg")
    )

    if not references or not queries:
        raise FileNotFoundError(
            "Reference or query images missing"
        )

    results = []

    for query in queries:
        for reference in references:
            result = compare_pair(
                reference, query, "output"
            )

            row = {
                "image_a": reference.name,
                "image_b": query.name,
                "label": "unknown",
                "pair_type": "unknown"
            }

            for field in FIELDS[4:]:
                row[field] = result[field]

            results.append(row)

    return results


def save_results(results, output_file):
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(results)


def print_metrics(results):
    known = [
        row for row in results
        if str(row["label"]).lower()
        in ("same", "different")
    ]

    if not known:
        print("Labels unknown: accuracy skipped.")
        return

    same = [
        row for row in known
        if row["label"].lower() == "same"
    ]

    different = [
        row for row in known
        if row["label"].lower() == "different"
    ]

    lookalike = [
        row for row in known
        if row["pair_type"].lower() == "lookalike"
    ]

    correct_same = sum(
        row["band"] == "SAME"
        for row in same
    )

    rejected_different = sum(
        row["band"] != "SAME"
        for row in different
    )

    false_lookalike = sum(
        row["band"] == "SAME"
        for row in lookalike
    )

    print("\nEVALUATION METRICS")

    print(
        f"Same recall: {correct_same}/{len(same)}"
    )

    print(
        "Different rejection:",
        f"{rejected_different}/{len(different)}"
    )

    print(
        "Lookalike false SAME:",
        f"{false_lookalike}/{len(lookalike)}"
    )

    print("\nRESULTS BY PAIR TYPE")

    types = sorted(set(
        row["pair_type"] for row in known
    ))

    for pair_type in types:
        group = [
            row for row in known
            if row["pair_type"] == pair_type
        ]

        correct = sum(
            (row["band"] == "SAME")
            if row["label"].lower() == "same"
            else (row["band"] != "SAME")
            for row in group
        )

        print(
            f"{pair_type}: {correct}/{len(group)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", required=True
    )

    parser.add_argument(
        "--pairs", default=None
    )

    parser.add_argument(
        "--out", default="results.csv"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)

    if args.pairs:
        results = evaluate_pairs(
            data_dir,
            args.pairs,
            args.out
        )
    else:
        results = evaluate_gallery(data_dir)

    save_results(results, args.out)

    print("\nEvaluation Completed!")
    print("Total comparisons:", len(results))
    print("Saved:", args.out)

    for band in ["SAME", "SIMILAR", "DIFFERENT"]:
        count = sum(
            row["band"] == band
            for row in results
        )
        print(f"{band}: {count}")

    print_metrics(results)


if __name__ == "__main__":
    main()

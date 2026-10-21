

import cv2
import numpy as np
import argparse
import time
from pathlib import Path


# Configuration
IMAGE_SIZE = 960
SIFT_FEATURES = 4000
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0

# Temporary thresholds: tune after testing all pairs
SAME_THRESHOLD = 40.0
SIMILAR_THRESHOLD = 15.0


def preprocess(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    height, width = image.shape[:2]
    scale = IMAGE_SIZE / max(height, width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    return resized, gray


def calculate_area(points, image_shape):
    if len(points) < 4:
        return 0.0

    x, y, w, h = cv2.boundingRect(
        np.float32(points)
    )

    image_height, image_width = image_shape[:2]

    area = (w * h) / (
        image_width * image_height
    )

    return float(area)


def match_images(path_a, path_b, output_dir):
    start_time = time.perf_counter()

    # Step 1: Image preprocessing
    image_a, gray_a = preprocess(path_a)
    image_b, gray_b = preprocess(path_b)

    # Step 2: SIFT feature extraction
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES
    )

    kp_a, des_a = sift.detectAndCompute(
        gray_a, None
    )

    kp_b, des_b = sift.detectAndCompute(
        gray_b, None
    )

    # Step 3: Feature matching
    good_matches = []
    inlier_matches = []
    total_matches = 0

    if (
        des_a is not None
        and des_b is not None
        and len(des_a) >= 1
        and len(des_b) >= 2
    ):
        matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=False
        )

        knn_matches = matcher.knnMatch(
            des_a,
            des_b,
            k=2
        )

        total_matches = len(knn_matches)

        # Step 4: Lowe's ratio test
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair

                if m.distance < LOWE_RATIO * n.distance:
                    good_matches.append(m)

    # Diagnostic measurements
    inlier_area_a = 0.0
    inlier_area_b = 0.0
    homography_valid = False

    # Step 5: RANSAC geometric verification
    if len(good_matches) >= 4:

        points_a = np.float32([
            kp_a[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        points_b = np.float32([
            kp_b[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        try:
            H, mask = cv2.findHomography(
                points_a,
                points_b,
                cv2.RANSAC,
                RANSAC_THRESHOLD
            )

            if (
                H is not None
                and mask is not None
                and np.all(np.isfinite(H))
            ):
                homography_valid = True

                # Keep only RANSAC verified matches
                inlier_matches = [
                    m
                    for m, valid in zip(
                        good_matches,
                        mask.ravel()
                    )
                    if valid
                ]

                if len(inlier_matches) >= 4:

                    verified_points_a = np.float32([
                        kp_a[m.queryIdx].pt
                        for m in inlier_matches
                    ])

                    verified_points_b = np.float32([
                        kp_b[m.trainIdx].pt
                        for m in inlier_matches
                    ])

                    inlier_area_a = calculate_area(
                        verified_points_a,
                        gray_a.shape
                    )

                    inlier_area_b = calculate_area(
                        verified_points_b,
                        gray_b.shape
                    )

        except cv2.error as error:
            print("RANSAC warning:", error)

    # Step 6: Calculate statistics
    inliers = len(inlier_matches)

    good_count = len(good_matches)

    inlier_ratio = (
        inliers / good_count
        if good_count > 0
        else 0.0
    )

    # Confidence from verified inliers
    if homography_valid and inliers >= 4:
        count_strength = (
            1 - np.exp(-inliers / 18.0)
        )

        confidence = (
            100.0
            * count_strength
            * inlier_ratio
        )
    else:
        confidence = 0.0

    confidence = float(
        np.clip(confidence, 0, 100)
    )

    # Initial classification thresholds
    if confidence >= SAME_THRESHOLD:
        band = "SAME"

    elif confidence >= SIMILAR_THRESHOLD:
        band = "SIMILAR"

    else:
        band = "DIFFERENT"

    # Step 7: Draw only RANSAC inlier matches
    result_image = cv2.drawMatches(
        image_a,
        kp_a,
        image_b,
        kp_b,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Step 8: Save output
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    name_a = Path(path_a).stem
    name_b = Path(path_b).stem

    output_path = (
        output_dir
        / f"{name_a}__{name_b}.jpg"
    )

    if not cv2.imwrite(
        str(output_path),
        result_image
    ):
        raise OSError(
            f"Unable to save output: {output_path}"
        )

    elapsed = (
        time.perf_counter() - start_time
    ) * 1000

    # Step 9: Print results
    print("\n========== IMAGE MATCHING RESULT ==========")

    print("Image A:", Path(path_a).name)
    print("Image B:", Path(path_b).name)

    print("-------------------------------------------")

    print("Keypoints A:", len(kp_a))
    print("Keypoints B:", len(kp_b))

    print("Total Matches:", total_matches)
    print("Good Matches:", good_count)

    print("RANSAC Inliers:", inliers)

    print(
        "Inlier Ratio:",
        round(inlier_ratio, 4)
    )

    print(
        "Inlier Area A:",
        round(inlier_area_a, 6)
    )

    print(
        "Inlier Area B:",
        round(inlier_area_b, 6)
    )

    print(
        "Homography Valid:",
        homography_valid
    )

    print("-------------------------------------------")

    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )

    print("Band:", band)

    print(
        "Processing Time:",
        round(elapsed, 2),
        "ms"
    )

    print("Saved:", output_path)

    print("===========================================\n")

    return {
        "keypoints_a": len(kp_a),
        "keypoints_b": len(kp_b),
        "total_matches": total_matches,
        "good_matches": good_count,
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "inlier_area_a": round(inlier_area_a, 6),
        "inlier_area_b": round(inlier_area_b, 6),
        "confidence": round(confidence, 2),
        "band": band,
        "time_ms": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(
        description="NMG Image Matching Challenge"
    )

    parser.add_argument(
        "--image-a",
        required=True,
        help="Reference image path"
    )

    parser.add_argument(
        "--image-b",
        required=True,
        help="Query image path"
    )

    parser.add_argument(
        "--out",
        default="output/",
        help="Output directory"
    )

    args = parser.parse_args()

    try:
        match_images(
            args.image_a,
            args.image_b,
            args.out
        )

    except (FileNotFoundError, OSError, cv2.error) as error:
        parser.exit(
            1,
            f"ERROR: {error}\n"
        )


if __name__ == "__main__":
    main()

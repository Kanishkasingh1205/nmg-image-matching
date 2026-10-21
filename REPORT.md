# NMG Labs — Image Matching Challenge

## Technical Report

### 1. Algorithm Selection

This project implements a classical computer vision pipeline using Python and OpenCV to determine whether two images depict the same physical stone object.

**Image preprocessing:** Images are resized to a fixed maximum dimension of 960 pixels and converted to grayscale.

**Feature extraction:** SIFT (Scale-Invariant Feature Transform) is used to detect local image features. It was selected for its robustness to scale and rotation changes.

**Feature matching:** Brute-Force Matcher (BFMatcher) with L2 distance is used to compare SIFT descriptors.

**Lowe's ratio test:** A ratio of 0.75 is used to remove ambiguous feature matches.

**Geometric verification:** RANSAC homography estimation with a 5-pixel reprojection threshold identifies geometrically consistent correspondences.

Only verified RANSAC inliers contribute to the confidence score.

### 2. Confidence Score

The confidence score is calculated using:

**Confidence = 100 × (1 − exp(−I / 18)) × (I / G)**

Where:
- I = Number of RANSAC inliers
- G = Number of good feature matches

If geometric verification fails, confidence is set to zero.

Classification thresholds:

| Classification | Confidence |
|---|---|
| SAME | 40% or higher |
| SIMILAR | 15% to below 40% |
| DIFFERENT | Below 15% |

These are development thresholds and are not guaranteed to generalize to unseen objects.

### 3. Evaluation Results

The supplied reference/query dataset contains 5 reference images and 20 query images.

A total of 100 reference-query comparisons were evaluated.

| Metric | Result |
|---|---|
| Total comparisons | 100 |
| Genuine SAME comparisons | 10 |
| Correct SAME predictions | 7 |
| Missed SAME predictions | 3 |
| Different comparisons | 90 |
| Correctly rejected different comparisons | 90 |
| False SAME predictions | 0 |
| SAME recall | 70% |
| Different rejection rate | 100% |
| Binary accuracy | 97% |

SIMILAR predictions are treated as not proven SAME for binary evaluation.

These results use labels inferred from the supplied development filenames, not an independent hidden test set.

### 4. Lowe's Ratio Experiment

Three Lowe's ratio thresholds were evaluated using the same development dataset.

| Ratio | Correct SAME | False SAME | Binary Accuracy |
|---|---|---|---|
| 0.70 | 7/10 | 0 | 97% |
| 0.75 | 7/10 | 0 | 97% |
| 0.80 | 7/10 | 2 | 95% |

The final ratio of 0.75 was selected because it retained zero false SAME predictions while providing the same SAME recall as 0.70.

### 5. Failure Analysis

**Failure 1 — Occlusion**

Query: `img_007__B01A_occlusion_same.jpg`

The same physical stone was classified as SIMILAR instead of SAME. The original test produced 8 RANSAC inliers and a confidence score of 19.14%.

Occlusion reduced the number of usable verified matches, resulting in insufficient evidence to pass the SAME threshold.

**Failure 2 — Viewpoint**

Query: `img_012__B03B_viewpoint_same.jpg`

This genuine SAME pair was classified as SIMILAR with 8 RANSAC inliers and a confidence score of 31.90%.

The viewpoint change likely reduced reliable descriptor correspondences and geometric verification strength.

**Additional challenge — False geometric matches**

A comparison between grey granite and an unrelated Carrara lookalike initially produced a false SAME classification with 18 RANSAC inliers and 43.76% confidence.

This demonstrated that RANSAC inlier count alone can be misleading when texture or match geometry is ambiguous. Later testing did not reproduce the false SAME prediction, so the final system should not be described as having conclusively solved this failure mode.

### 6. Limitations and Future Improvements

- Improve feature detection for heavily occluded and low-texture stone surfaces.
- Evaluate CLAHE enhancement against the grayscale baseline.
- Introduce validated spatial-distribution and homography-plausibility checks.
- Compare SIFT with ORB on the same dataset.
- Cache image descriptors for faster batch evaluation.
- Validate thresholds using unseen, independently labelled objects.

### 7. Conclusion

The system successfully implements SIFT feature extraction, Lowe's ratio filtering, RANSAC geometric verification, and inlier-based confidence scoring.

On the supplied development data, it achieved 97% binary accuracy, 70% SAME recall, and zero false SAME predictions at a Lowe's ratio of 0.75.

The remaining challenges involve recovering genuine matches under difficult imaging conditions while avoiding false acceptance of lookalike objects.
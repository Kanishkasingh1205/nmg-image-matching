# NMG Image Matching Challenge

## Project Overview

This project implements a classical computer vision image-matching system using Python and OpenCV.

The system determines whether two photographs show the same physical object based on local feature correspondences and geometric verification.

## Technologies Used

- Python 3.13
- OpenCV 5.0
- NumPy
- SIFT Feature Detection
- BFMatcher
- Lowe's Ratio Test
- RANSAC Homography

## Algorithm

1. Load and resize images.
2. Convert images to grayscale.
3. Extract SIFT keypoints and descriptors.
4. Match descriptors using BFMatcher with L2 distance.
5. Apply Lowe's ratio test (0.75).
6. Perform geometric verification using RANSAC.
7. Calculate confidence using verified inliers.
8. Classify as SAME, SIMILAR, or DIFFERENT.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Compare Two Images

```bash
python match.py --image-a data/images/img_001.jpg --image-b data/images/img_002.jpg --out output/
```

Replace the image paths with files available in your dataset.

## Evaluate the Dataset

For the supplied reference/query dataset:

```bash
python evaluate.py --data Intern_Assignment_Dataset --out results.csv
```

For a labelled pairs CSV:

```bash
python evaluate.py --data data --pairs data/pairs.csv --out results.csv
```

## Development Dataset Results

- Total comparisons: 100
- Genuine SAME pairs: 10
- Correct SAME predictions: 7
- Different pairs rejected: 90 out of 90
- False SAME predictions: 0
- Binary classification accuracy: 97%
- SAME recall: 70%

These figures were calculated using filename-derived labels from the supplied development dataset. They do not represent hidden-set performance. SIMILAR is counted as not proven SAME.

## Limitations

Low-texture regions, repeated patterns, heavy occlusion, and challenging viewpoint changes can reduce matching reliability. Thresholds were tested on the provided development data and may not generalize to unseen images.

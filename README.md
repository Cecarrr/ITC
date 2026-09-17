# ITC — AI & Machine Learning Division Projects

Computer vision projects built during my time on ITC's AI & Machine Learning
division. All models trained with Ultralytics YOLO (classification, object
detection, and segmentation variants) on datasets sourced or provided via
Roboflow.

## Projects

| Project | Task | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| [Waste Materials Classification](./Computer%20Vision/waste-classification) | 6-class classification | 94.2%¹ | 94.1%¹ | N/A² | N/A² |
| [Chess Piece Object Detection](./Computer%20Vision/chess-detection) | 12-class object detection | 94.2% | 87.8% | 91.3% | 66.0% |
| [Hair Segmentation](./Computer%20Vision/hair-segmentation) | Binary segmentation (mask) | 96.9% | 90.9% | 95.5% | 75.6% |
| [Keypoint Detection](./Computer%20Vision/keypoint-detection) | MediaPipe Pose Landmarker keypoints | N/A | N/A | N/A | N/A |

¹ Classification metrics: test accuracy and macro-averaged per-class recall
(precision/recall per class, not detection-style precision). <br>
² mAP is a detection/segmentation metric and doesn't apply to classification.

Each project folder has its own README with dataset, approach, full results,
and known limitations.

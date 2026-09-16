# Waste Material Classification

Image classification model that sorts waste into 6 categories: biodegradable,
cardboard, glass, metal, paper, and plastic.

## Context
Built as part of ITC's AI & Machine Learning division work.

## Dataset
[Waste Classifier — Roboflow Universe](https://universe.roboflow.com/caesar-ylvsf/waste-classifier-backup-pjrq3)
- 8,650 training images
- 2,481 validation images
- 1,219 test images

## Approach
- **Model**: YOLO26n-cls (Ultralytics)
- **Input size**: 320×320
- **Training**: 50 epochs, batch size 32, AdamW optimizer, cosine LR schedule
  (lr0=0.001), early stopping (patience=8)
- **Regularization**: label smoothing (0.2), dropout (0.15), weight decay (0.0006)
  — used to reduce overfitting on a moderately sized dataset
- This configuration was selected after comparing multiple data sources and
  hyperparameter settings; the results below reflect the best-performing run

## Results
Test set Top-1 accuracy: **94.18%**

| Class         | Recall |
|---------------|-------:|
| Biodegradable | 97.5%  |
| Glass         | 95.1%  |
| Cardboard     | 94.7%  |
| Metal         | 93.5%  |
| Paper         | 93.2%  |
| Plastic       | 90.5%  |

Full confusion matrices: [`confusion_matrix.png`](./result/confusion_matrix.png),
[`confusion_matrix_normalized.png`](./result/confusion_matrix_normalized.png)

## Limitations
- Plastic is the weakest class, most often confused with paper and glass
  likely due to visual similarity in some material textures/lighting in the
  dataset.
- Labels weren't verified image-by-image. An earlier candidate dataset was
  discarded after spot checks revealed inconsistent/incorrect labeling; the
  dataset used here was judged reliable based on that comparison and the
  gap in resulting performance.
- Earlier experiment runs (different sources/hyperparameters) weren't kept
  or logged and only the final configuration is documented here.

## Exported formats
Model exported for deployment across platforms:
- `onnx` — desktop, server, web, C++
- `tflite` — Android, Raspberry Pi, embedded/IoT
- `ncnn` — Android
- `coreml` — iOS, macOS, Vision Pro, Apple Watch
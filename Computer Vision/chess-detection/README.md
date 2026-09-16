# Chess Piece Object Detection

Detects and classifies chess pieces on a board — 12 classes (black/white ×
king, queen, bishop, knight, rook, pawn).

## Context
Built as part of ITC's AI & Machine Learning division work.

## Dataset
[Chess piece dataset — Roboflow Universe](https://universe.roboflow.com/caesar-ylvsf/sp2-ym1iq-5qvbo)
- 6,863 training images
- 1,964 validation images
- 982 test images

## Approach
- **Model**: YOLO26n (detection), input size 640×640 — raised from the
  320px used in the waste classification project, since chess pieces
  (especially pawns) are small objects that need more resolution to detect
- **Training**: up to 50 epochs (early stopping patience=8), batch size 32,
  AdamW optimizer, cosine LR schedule (lr0=0.001), weight decay 0.0006
- This configuration was selected after comparing dataset versions; two
  earlier dataset attempts were discarded due to poor camera angle/lighting
  or incomplete piece labeling

## Results
Test set (982 images, 21,326 instances):

| Metric    | Score  |
|-----------|-------:|
| Precision | 94.2%  |
| Recall    | 87.8%  |
| mAP50     | 91.3%  |
| mAP50-95  | 66.0%  |

Per-class breakdown:

| Class         | Precision | Recall | mAP50 | mAP50-95 |
|---------------|----------:|-------:|------:|---------:|
| Black bishop  | 94.1%     | 91.2%  | 93.3% | 63.8%    |
| Black king    | 91.3%     | 87.6%  | 91.4% | 70.0%    |
| Black knight  | 93.8%     | 85.2%  | 89.0% | 63.3%    |
| Black pawn    | 96.0%     | 88.3%  | 92.5% | 63.0%    |
| Black queen   | 92.2%     | 84.1%  | 90.2% | 69.5%    |
| Black rook    | 93.0%     | 84.8%  | 88.2% | 65.0%    |
| White bishop  | 95.5%     | 90.2%  | 93.0% | 64.9%    |
| White king    | 93.3%     | 90.7%  | 94.2% | 72.2%    |
| White knight  | 96.3%     | 87.0%  | 90.4% | 64.6%    |
| White pawn    | 96.2%     | 88.2%  | 91.9% | 61.6%    |
| White queen   | 94.4%     | 88.4%  | 92.9% | 70.6%    |
| White rook    | 94.4%     | 87.3%  | 88.7% | 64.0%    |

Black-side pieces (particularly queen, knight, rook) show slightly lower
recall than their white counterparts — possibly a lighting/contrast effect,
since darker piece material reflects light differently.

Full confusion matrices: [`confusion_matrix.png`](./result/confusion_matrix.png),
[`confusion_matrix_normalized.png`](./result/confusion_matrix_normalized.png)

Confusion matrices are from this same test evaluation (982 images, 21,326 instances) - background
row/column counts are in the same order of magnitude as the class counts here
(unlike a training-time validation confusion matrix, where background counts
can look inflated due to how YOLO scores anchor-level background at a
different granularity).

## Limitations
- Piece classes with similar silhouettes (king/queen, bishop/rook) are more
  often confused, particularly among the black pieces
- Trained only on classic-style piece sets which likely to perform worse on
  fantasy/unusual/non-standard piece designs
- Lighting and material sensitivity: pieces are partly distinguished by how
  light reflects off their surface, so different lighting conditions can
  affect detection even for the same piece type
- Testing wasn't extensive due to time constraints; the final dataset was
  chosen based on a qualitative comparison against a clearly worse earlier
  version (bad angles/lighting, incomplete labeling), not a systematic
  per-image label audit
- Two earlier dataset and run attempts weren't kept or logged, only the final
  configuration is documented here

## Exported formats
Model exported for deployment across platforms:
- `onnx` — desktop, server, web, C++
- `tflite` — Android, Raspberry Pi, embedded/IoT
- `ncnn` — Android
- `coreml` — iOS, macOS, Vision Pro, Apple Watch
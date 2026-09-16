# Hair Segmentation

Pixel-level segmentation of hair regions in images (binary: hair vs. background).

## Context
Built as part of ITC's AI & Machine Learning division work. This project was
a given competitive assignment — the dataset was provided, and division
members compared results on the same data rather than each sourcing their own.

## Dataset
[Hair segmentation dataset — Roboflow Universe](https://universe.roboflow.com/caesar-ylvsf/hair-2pjwo-ng8p4)
- ~1,894 training images
- ~796 validation images (after excluding a small number of corrupt/malformed label files that Ultralytics flagged automatically)
- 479 images

## Approach
- **Model**: YOLO26n-seg (Ultralytics), input size 640×640
- **Training**: 50 epochs, batch size 32, AdamW optimizer, cosine LR schedule
  (lr0=0.001), weight decay 0.0006
- **Segmentation-specific settings**: `mask_ratio=2` (raised from the default
  4, for finer mask precision on a detailed object like hair) and
  `retina_masks=True` (smoother mask output)
- Hit the best result on the first full run; later attempts adjusting
  hyperparameters didn't improve overall performance — they traded precision
  for recall (or vice versa) rather than raising both

## Results
Test set (479 images, 550 instances):

| Metric    | Box   | Mask  |
|-----------|------:|------:|
| Precision | 96.9% | 96.9% |
| Recall    | 90.7% | 90.9% |
| mAP50     | 96.2% | 95.5% |
| mAP50-95  | 84.7% | 75.6% |

Full confusion matrices: [`confusion_matrix.png`](./result/confusion_matrix.png),
[`confusion_matrix_normalized.png`](./result/confusion_matrix_normalized.png)

For reference, training-time validation (796 images, 900 instances) scored
close to these numbers (Box P 95.2%/R 92.9%, Mask P 93.6%/R 93.7%), and the
confusion matrix from that validation run showed ~95% recall on the hair
class with 92 background regions misclassified as hair which is consistent with
the background-bleed limitation below.

## Limitations
- **Lighting sensitivity**: insufficient or low lighting causes detection to
  become unstable, the model can flicker on and off across frames/nearby
  images under the same conditions
- **Background bleed**: when the background is a similar shade to the
  subject's hair and appears visually connected in-frame, the model tends to
  extend the hair mask into the background rather than separating them
- Only one full training configuration was properly explored; hyperparameter
  adjustments after the first run traded metrics against each other (e.g.
  precision up, recall down) rather than improving both, suggesting this
  configuration may be close to the ceiling for this model/dataset without a
  different approach (e.g. more data, augmentation targeting lighting variance)
- A small number of training/validation labels were corrupted (mixed
  segmentation and detection format) and were automatically excluded rather
  than manually reviewed and fixed

## Exported formats
Model exported for deployment across platforms:
- `onnx` — desktop, server, web, C++
- `tflite` — Android, Raspberry Pi, embedded/IoT
- `ncnn` — Android
- `coreml` — iOS, macOS, Vision Pro, Apple Watch
# Desk Posture Monitor (Keypoint Detection)

Real-time desk posture checker built on MediaPipe Pose Landmarker keypoints it flags sustained slouching and lateral lean from a normal chest-up, this will need laptop/desk webcam feed (not a full-body external camera).

## Context
Built as part of ITC's AI & Machine Learning division work.

## Dataset
N/A. 
<br> This project doesn't train a model, it runs entirely on MediaPipe's
pretrained Pose Landmarker (`pose_landmarker_lite`, auto-downloaded from
Google's hosted model storage at first run) and builds rule-based logic on
top of the 33 keypoints it returns per frame. No custom dataset was
collected or needed.

## Approach
- **Model**: MediaPipe Pose Landmarker (lite variant), running in `VIDEO`
  mode over live webcam frames with 33 body keypoints per frame, each with an
  (x, y) position and a visibility confidence score
- **Two independent heuristic signals** derived from keypoints:
  - *Neck angle*: angle between the shoulder-midpoint and ear-midpoint vs.
    vertical who picks up the head tilting forward
  - *Nose-shoulder ratio*: vertical gap between the nose and shoulder
    midpoint, normalized by shoulder width who picks up the head dropping
    toward the shoulder line, independent of the angle signal
- **Three-zone system** (Frontal / Turning / Profile), based on live
  shoulder width vs a calibrated baseline with hysteresis on the zone
  boundaries to avoid flicker. Turning your head/body shrinks apparent
  shoulder width through foreshortening well before you're in full profile,
  which breaks the nose-ratio's width normalization so each zone only
  trusts the signals that are actually still reliable in it (see table in
  Results)
- **Fail-safe combination logic**: flags slouching if either signal alone
  crosses its full threshold, or both cross a lower joint threshold
  together; falls back to whichever single signal is available if the
  other is occluded; reports "no reliable signal" instead of guessing if
  neither is
- **Calibration step** (press `c`) instead of hardcoded thresholds, since
  baseline posture and camera setup vary per person/device
- **Visibility gating** on every keypoint (drops low-confidence landmarks
  rather than trusting them), **duration debounce** before any alert fires,
  and **rolling-median smoothing** to reduce frame-to-frame jitter

### Setup & usage
```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python posture_monitor.py
```
Face the camera, sit up straight, press `c` to calibrate, then work
normally until sustained deviation triggers an on-screen slouch warning.
Press `q` to quit. This needs live webcam access, so it's meant to run
locally and not in a notebook environment.

## Results
No held-out test set or precision/recall metrics since this isn't a trained
classifier, it's a real-time rule-based system tuned through iterative live
testing on one subject. Testing outcomes at current thresholds:

| Scenario | Outcome |
|---|---|
| Full slouch vs. slight forward lean | Reliably separated (~20° neck-angle deviation vs. ~3–9°) |
| Reaching up to grab something | No longer false-triggers slouch/tilt, after adding duration debounce (very rare false trigger remains) |
| Turning to the side / partial turn | No longer flickers between states, after replacing a binary frontal/side split with the three-zone hysteresis system |
| Looking down at keyboard | Does not trigger |
| Looking down at phone/floor for a sustained period | Occasionally flagged as slouching — see Limitations |
| Fully facing away from camera | Correctly reports "no reliable signal" rather than guessing |

## Limitations
- **Can't distinguish desk-slouch from deliberately looking down** - a
  sustained downward head tilt looks geometrically identical whether it's
  bad desk posture or checking a phone/something on the floor. This is a
  limitation of using pose geometry alone, not a bug.
- **No signal when fully facing away from the camera** - with no face
  keypoints visible, the system reports "no reliable signal" rather than
  attempting a guess. Expected behavior, not a failure case.
- **Thresholds are tuned for one tester's setup** (camera distance, own
  posture habits) - a different user/camera should recalibrate and may
  need to adjust the constants at the top of `posture_monitor.py`. Setting
  thresholds unusually loose is a configuration choice on the user's end,
  not a code issue.
- Testing was iterative and manual (live sessions, not a labeled
  benchmark set), so the outcomes above reflect one person's setup and
  usage patterns rather than a systematic evaluation.

## Exported formats
N/A. 
<br> Nothing is trained or exported in this project since it runs directly
against MediaPipe's stock pose landmarker model at inference time.

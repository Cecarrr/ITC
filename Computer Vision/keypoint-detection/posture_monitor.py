"""
Posture Monitor v4 - real-time desk posture checker using MediaPipe Pose.

v4 change: replaces the binary frontal/side-view split with a three-zone
system (frontal / turning / profile) that uses hysteresis to avoid
flickering between zones near a boundary, and scopes which signals are
trusted per zone:

- FRONTAL: shoulder width close to calibrated baseline -> both neck angle
  and nose-shoulder ratio are trustworthy, shoulder-tilt check runs too.
- TURNING: shoulder width has shrunk from foreshortening (mid-turn) ->
  the nose-ratio's shoulder-width normalization is no longer reliable, so
  it's dropped. Neck angle (a pure angle, not width-dependent) still runs.
  Shoulder-tilt is suppressed.
- PROFILE: shoulder width has shrunk too far to trust anything scale- or
  symmetry-dependent -> reports "no reliable signal" instead of guessing.

Design assumption carried from v3: a typical chest-up laptop/desk webcam
framing (not a full-body external camera), so no hip landmarks.

Two independent slouch signals (used depending on zone, see above):
1. Neck angle: angle between shoulder-midpoint and ear-midpoint vs vertical.
2. Nose-shoulder ratio: vertical gap between nose and shoulder line,
   normalized by shoulder width.

Known limitation (see README): this can't distinguish "slouching at the
desk" from "genuinely looking down at your phone/something on the floor"
for a sustained period -- both look geometrically identical to the model.

Usage:
1. Run the script. A webcam window opens with landmarks drawn on top.
2. Sit up straight, face the camera, press 'c' to calibrate baseline.
3. Keep working normally -- sustained deviation triggers a slouch warning.
4. Press 'q' to quit.

First run auto-downloads the MediaPipe pose_landmarker model (~30MB).
"""

import os
import time
import math
import urllib.request
import statistics
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

SLOUCH_ANGLE_THRESHOLD = 10.0       # neck-angle deviation (deg) -> slouch
NOSE_RATIO_DROP_THRESHOLD = 0.20    # nose-shoulder ratio drop -> slouch
SLOUCH_TIME_THRESHOLD = 5.0         # seconds sustained before alerting

SHOULDER_TILT_THRESHOLD_PX = 25
SHOULDER_TILT_TIME_THRESHOLD = 2.0

VISIBILITY_THRESHOLD = 0.5          # MediaPipe visibility score cutoff (0-1)
SMOOTHING_WINDOW = 8                 # frames used for rolling median

# --- zone boundaries, as a fraction of calibrated frontal shoulder width ---
# Hysteresis: each zone has an "enter" ratio and an "exit" ratio a bit
# further out, so sitting near a boundary doesn't flicker between zones
# every frame.
FRONTAL_ENTER_RATIO = 0.85   # need ratio >= this to (re-)enter FRONTAL
FRONTAL_EXIT_RATIO = 0.75    # once frontal, only leave when ratio < this
PROFILE_ENTER_RATIO = 0.45   # ratio below this -> PROFILE
PROFILE_EXIT_RATIO = 0.55    # once profile, only leave when ratio >= this
# Anything between the exit/enter bands above = TURNING zone.

# Pose landmark indices (MediaPipe Pose has 33 landmarks total)
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_EAR, RIGHT_EAR = 7, 8

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def visible_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    if lm.visibility is not None and lm.visibility < VISIBILITY_THRESHOLD:
        return None
    return (lm.x * w, lm.y * h)


def midpoint_or_fallback(p_left, p_right):
    if p_left is not None and p_right is not None:
        return ((p_left[0] + p_right[0]) / 2, (p_left[1] + p_right[1]) / 2)
    if p_left is not None:
        return p_left
    if p_right is not None:
        return p_right
    return None


def angle_from_vertical(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def get_neck_angle(landmarks, w, h):
    l_sh = visible_point(landmarks, LEFT_SHOULDER, w, h)
    r_sh = visible_point(landmarks, RIGHT_SHOULDER, w, h)
    l_ear = visible_point(landmarks, LEFT_EAR, w, h)
    r_ear = visible_point(landmarks, RIGHT_EAR, w, h)

    shoulder_pt = midpoint_or_fallback(l_sh, r_sh)
    ear_pt = midpoint_or_fallback(l_ear, r_ear)

    if shoulder_pt is None or ear_pt is None:
        return None
    return angle_from_vertical(shoulder_pt, ear_pt)


def get_shoulder_distance(landmarks, w, h):
    l_sh = visible_point(landmarks, LEFT_SHOULDER, w, h)
    r_sh = visible_point(landmarks, RIGHT_SHOULDER, w, h)
    if l_sh is None or r_sh is None:
        return None
    return math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])


def get_nose_shoulder_ratio(landmarks, w, h):
    nose = visible_point(landmarks, NOSE, w, h)
    l_sh = visible_point(landmarks, LEFT_SHOULDER, w, h)
    r_sh = visible_point(landmarks, RIGHT_SHOULDER, w, h)

    if nose is None or l_sh is None or r_sh is None:
        return None

    shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    shoulder_width = math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
    if shoulder_width < 1e-3:
        return None

    vertical_gap = shoulder_mid[1] - nose[1]
    return vertical_gap / shoulder_width


def get_shoulder_tilt(landmarks, w, h):
    l_sh = visible_point(landmarks, LEFT_SHOULDER, w, h)
    r_sh = visible_point(landmarks, RIGHT_SHOULDER, w, h)
    if l_sh is None or r_sh is None:
        return None
    return abs(l_sh[1] - r_sh[1])


def update_zone(current_zone, width_ratio):
    """
    Hysteresis state machine over three zones: 'frontal', 'turning', 'profile'.
    width_ratio is live shoulder distance / calibrated frontal shoulder distance,
    or None if it can't currently be measured (treated as worst case: profile).
    """
    if width_ratio is None:
        return "profile"

    if current_zone == "frontal":
        if width_ratio < PROFILE_ENTER_RATIO:
            return "profile"
        if width_ratio < FRONTAL_EXIT_RATIO:
            return "turning"
        return "frontal"

    if current_zone == "turning":
        if width_ratio >= FRONTAL_ENTER_RATIO:
            return "frontal"
        if width_ratio < PROFILE_ENTER_RATIO:
            return "profile"
        return "turning"

    # current_zone == "profile"
    if width_ratio >= FRONTAL_ENTER_RATIO:
        return "frontal"
    if width_ratio >= PROFILE_EXIT_RATIO:
        return "turning"
    return "profile"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera index/permissions.")

    baseline_neck_angle = None
    baseline_nose_ratio = None
    baseline_shoulder_distance = None

    slouch_start_time = None
    tilt_start_time = None
    frame_timestamp_ms = 0
    current_zone = "frontal"

    neck_hist = deque(maxlen=SMOOTHING_WINDOW)
    ratio_hist = deque(maxlen=SMOOTHING_WINDOW)
    tilt_hist = deque(maxlen=SMOOTHING_WINDOW)

    print("Press 'c' to calibrate your good posture baseline (face the camera). "
          "Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        status_text = "No person detected"
        status_color = (0, 0, 255)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            neck_angle = get_neck_angle(landmarks, w, h)
            nose_ratio = get_nose_shoulder_ratio(landmarks, w, h)
            shoulder_dist = get_shoulder_distance(landmarks, w, h)
            shoulder_tilt = get_shoulder_tilt(landmarks, w, h)

            if neck_angle is not None:
                neck_hist.append(neck_angle)
            if nose_ratio is not None:
                ratio_hist.append(nose_ratio)
            if shoulder_tilt is not None:
                tilt_hist.append(shoulder_tilt)

            smooth_neck = statistics.median(neck_hist) if neck_hist else None
            smooth_ratio = statistics.median(ratio_hist) if ratio_hist else None
            smooth_tilt = statistics.median(tilt_hist) if tilt_hist else None

            # --- update zone (hysteresis) ---
            width_ratio = None
            if baseline_shoulder_distance is not None and shoulder_dist is not None:
                width_ratio = shoulder_dist / baseline_shoulder_distance
            current_zone = update_zone(current_zone, width_ratio)

            for idx in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR, NOSE):
                pt = visible_point(landmarks, idx, w, h)
                if pt is not None:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), -1)

            if baseline_neck_angle is None and baseline_nose_ratio is None:
                status_text = "Not calibrated - face camera, sit upright, press 'c'"
                status_color = (0, 165, 255)
            elif current_zone == "profile":
                slouch_start_time = None
                status_text = "No reliable signal (side profile)"
                status_color = (150, 150, 150)
            else:
                neck_dev = None
                if smooth_neck is not None and baseline_neck_angle is not None:
                    neck_dev = smooth_neck - baseline_neck_angle

                # nose-ratio signal only trusted in the FRONTAL zone -- in
                # TURNING, foreshortened shoulder width makes its
                # normalization unreliable.
                ratio_drop = None
                if current_zone == "frontal":
                    if smooth_ratio is not None and baseline_nose_ratio is not None:
                        ratio_drop = baseline_nose_ratio - smooth_ratio

                conditions = []
                if neck_dev is not None:
                    conditions.append(neck_dev > SLOUCH_ANGLE_THRESHOLD)
                if ratio_drop is not None:
                    conditions.append(ratio_drop > NOSE_RATIO_DROP_THRESHOLD)
                if neck_dev is not None and ratio_drop is not None:
                    conditions.append(
                        neck_dev > SLOUCH_ANGLE_THRESHOLD * 0.6
                        and ratio_drop > NOSE_RATIO_DROP_THRESHOLD * 0.6
                    )

                if not conditions:
                    status_text = "No reliable signal (face/shoulders not visible)"
                    status_color = (0, 0, 255)
                else:
                    is_slouching_now = any(conditions)

                    if is_slouching_now:
                        if slouch_start_time is None:
                            slouch_start_time = time.time()
                        elapsed = time.time() - slouch_start_time
                        if elapsed >= SLOUCH_TIME_THRESHOLD:
                            status_text = f"SLOUCHING for {elapsed:.0f}s - sit up!"
                            status_color = (0, 0, 255)
                        else:
                            status_text = "Slouch detected..."
                            status_color = (0, 140, 255)
                    else:
                        slouch_start_time = None
                        status_text = "Posture OK"
                        status_color = (0, 200, 0)

                    label_parts = [f"Zone: {current_zone}"]
                    if smooth_neck is not None and baseline_neck_angle is not None:
                        label_parts.append(f"Neck: {smooth_neck:.1f}/{baseline_neck_angle:.1f}")
                    if ratio_drop is not None:
                        label_parts.append(f"NoseRatio: {smooth_ratio:.2f}/{baseline_nose_ratio:.2f}")
                    cv2.putText(frame, "  ".join(label_parts), (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # --- shoulder tilt: only checked in FRONTAL zone ---
            if current_zone != "frontal":
                tilt_start_time = None
                cv2.putText(frame, f"Tilt check paused ({current_zone})", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            elif smooth_tilt is not None:
                if smooth_tilt > SHOULDER_TILT_THRESHOLD_PX:
                    if tilt_start_time is None:
                        tilt_start_time = time.time()
                    tilt_elapsed = time.time() - tilt_start_time
                    if tilt_elapsed >= SHOULDER_TILT_TIME_THRESHOLD:
                        cv2.putText(frame, f"Shoulder tilt for {tilt_elapsed:.0f}s",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                else:
                    tilt_start_time = None

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2)

        cv2.imshow("Posture Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c') and result.pose_landmarks:
            lm = result.pose_landmarks[0]
            n = get_neck_angle(lm, w, h)
            r = get_nose_shoulder_ratio(lm, w, h)
            d = get_shoulder_distance(lm, w, h)

            calibrated_any = False
            if n is not None:
                baseline_neck_angle = n
                calibrated_any = True
            if r is not None:
                baseline_nose_ratio = r
                calibrated_any = True
            if d is not None:
                baseline_shoulder_distance = d

            slouch_start_time = None
            tilt_start_time = None
            current_zone = "frontal"
            neck_hist.clear()
            ratio_hist.clear()
            tilt_hist.clear()

            if calibrated_any:
                print(
                    f"Calibrated. neck_angle={baseline_neck_angle}, "
                    f"nose_ratio={baseline_nose_ratio}, "
                    f"shoulder_dist={baseline_shoulder_distance}"
                )
            else:
                print("Calibration failed - no reliable landmarks. "
                      "Face the camera and make sure your face/shoulders are visible.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

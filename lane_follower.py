#!/usr/bin/env python3
"""
Picar-X Two-Lane Track Follower with Traffic Sign Handling

Track layout (looking from above):
  [LEFT BOUNDARY solid] [left lane] [CENTER DASHES] [right lane] [RIGHT BOUNDARY solid]

Car drives in the RIGHT lane by default.
  - Camera left side: center dashes
  - Camera right side: right boundary (solid)
  - Target: keep lane midpoint at camera center

Sensors used:
  - Camera (Picamera2): mid-range lane detection + sign detection (upper frame)
  - Grayscale (A0/A1/A2): fast boundary-crossing alerts
  - Ultrasonic: obstacle stop

Usage:
  python3 lane_follower.py
  python3 lane_follower.py --left-lane      # drive in left lane instead
  python3 lane_follower.py --save-frames    # save debug frames to /tmp/lf_debug/
  python3 lane_follower.py --no-signs       # disable traffic sign detection
"""

import argparse
import os
import threading
import time
from time import monotonic, sleep

import cv2
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx

# ── Hardware init ────────────────────────────────────────────────────────────
px = Picarx()

# ════════════════════════════════════════════════════════════════════════════
#  TUNABLE CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Camera
CAM_W, CAM_H = 640, 480
CAM_FPS      = 30
CAM_TILT     = -15.0
CAM_PAN      =   0.0

# White detection (HSV)
WHITE_V_MIN  = 175      # lower than before — foam mat is dark, tape is bright
WHITE_S_MAX  =  60
ADAPTIVE_K   =  0.20

# ROI zones (fraction of frame height)
LANE_ROI_TOP  = 0.42   # lane following: mid-lower frame at -15° tilt
LANE_ROI_BOT  = 0.82
SIGN_ROI_TOP  = 0.00   # sign detection: upper portion of frame
SIGN_ROI_BOT  = 0.45

# Lane detection
MIN_CONTOUR_AREA = 200     # lower threshold — lines are small when viewed from distance
MIN_ASPECT_RATIO = 0.8     # accept both tape strips and dashes (not just horizontal)

# Grayscale thresholds — raw ADC values
# Dark mat ≈ 200–400,  White tape ≈ 800–2000
# If these are wrong, run: python3 picar-x/example/1.cali_grayscale.py
GS_WHITE_THRESHOLD = 600   # ADC value above which sensor is "on white"

# PID (gentler than original to stop oscillation)
KP = 22.0
KI =  0.15
KD =  7.0

# Steering
MAX_STEER  = 25.0   # hard servo limit (degrees)
STEER_RATE =  7.0   # max change per frame — gentle, prevents oscillation

# Speeds (%)
SPEED_CRUISE  = 14
SPEED_CORRECT = 10
SPEED_CREEP   =  7
SPEED_TURN    = 10

# Obstacle
STOP_DIST =  15   # cm — stop if object closer than this
SLOW_DIST =  40   # cm

# Gap handling (between dashes on centre line)
GAP_MAX_FRAMES = 12      # hold last steer this many frames before seeking
GAP_HOLD_DECAY =  0.97   # per-frame decay of last known lane error

# Loop
LOOP_HZ    = 25
PRINT_EVERY = 8

# Traffic sign
SIGN_INTERVAL   = 0.40   # seconds between TFLite inferences
SIGN_CONFIDENCE = 70     # minimum % confidence to act on a sign
SIGN_MIN_AREA   =  900   # min bounding-box area (px²) to consider a detection
SIGN_ARM_FRAMES =  20    # consecutive frames sign must be seen before arming a turn
STOP_HOLD_SEC   =  3.0   # seconds to stay stopped at a stop sign
TURN_HOLD_SEC   =  2.2   # seconds to hold turn steer before resuming lane follow


# ════════════════════════════════════════════════════════════════════════════
#  LANE DETECTOR
# ════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Detects the two boundaries of the car's current lane from a camera frame.

    Returns a normalised error:
      0.0  = car is centred in lane
      +1.0 = car is too far RIGHT (must steer left)
      -1.0 = car is too far LEFT  (must steer right)
    """

    def __init__(self, drive_left_lane: bool = False):
        self._drive_left    = drive_left_lane
        self.last_error_valid = False  # True when previous detect() found lines

    def _white_mask(self, roi_bgr):
        hsv    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean_v = float(hsv[:, :, 2].mean())
        v_min  = int(WHITE_V_MIN + ADAPTIVE_K * (mean_v - 128))
        v_min  = max(100, min(245, v_min))
        mask   = cv2.inRange(
            hsv,
            np.array([0,           0, v_min], np.uint8),
            np.array([180, WHITE_S_MAX,  255], np.uint8),
        )
        k    = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask, v_min

    def detect(self, frame):
        """
        Returns (error, total_px, debug_info_dict, annotated_frame).
        error is None if no lines found.
        """
        h, w = frame.shape[:2]
        y0   = int(h * LANE_ROI_TOP)
        y1   = int(h * LANE_ROI_BOT)
        roi  = frame[y0:y1, :]

        mask, v_thr = self._white_mask(roi)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect all significant white blobs with their centroids
        blobs = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if aspect < MIN_ASPECT_RATIO:
                continue
            blobs.append({'cx': cx, 'cy': cy, 'area': area, 'cnt': cnt,
                          'x': x, 'y': y, 'w': bw, 'h': bh})

        ann = frame.copy()
        cv2.rectangle(ann, (0, y0), (w-1, y1), (0, 200, 200), 1)
        cv2.line(ann,      (w//2, y0), (w//2, y1), (255, 0, 0), 1)

        total_px = sum(b['area'] for b in blobs)

        if not blobs:
            cv2.putText(ann, 'NO LINES', (10, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self.last_error_valid = False
            return None, 0, {'v_thr': v_thr, 'blobs': 0}, ann

        # ── Separate blobs into LEFT half and RIGHT half of frame ──────────
        left_blobs  = [b for b in blobs if b['cx'] < w / 2]
        right_blobs = [b for b in blobs if b['cx'] >= w / 2]

        def weighted_cx(blob_list):
            if not blob_list:
                return None
            total_a = sum(b['area'] for b in blob_list)
            return sum(b['cx'] * b['area'] for b in blob_list) / total_a

        left_cx  = weighted_cx(left_blobs)
        right_cx = weighted_cx(right_blobs)

        # ── Compute lane centre ────────────────────────────────────────────
        if left_cx is not None and right_cx is not None:
            lane_cx = (left_cx + right_cx) / 2.0
        elif left_cx is not None:
            # Only left line visible → infer right side
            lane_cx = (left_cx + w) / 2.0
        elif right_cx is not None:
            # Only right line visible → infer left side
            lane_cx = right_cx / 2.0
        else:
            self.last_error_valid = False
            return None, 0, {'v_thr': v_thr, 'blobs': 0}, ann

        # For left-lane driving, mirror the logic
        if self._drive_left:
            lane_cx = w - lane_cx

        # Normalised error: positive = car is right of lane centre → steer left
        error = (lane_cx - w / 2.0) / (w / 2.0)

        # ── Annotate ──────────────────────────────────────────────────────
        for b in blobs:
            cnt_abs = b['cnt'] + np.array([0, y0])
            color   = (0, 255, 100) if b['cx'] < w/2 else (0, 100, 255)
            cv2.drawContours(ann, [cnt_abs], -1, color, 1)
            cv2.circle(ann, (int(b['cx']), int(b['cy']) + y0), 5, (0, 0, 255), -1)

        lane_px = int(lane_cx)
        cv2.line(ann, (lane_px, y0), (lane_px, y1), (0, 255, 255), 2)

        if left_cx  is not None:
            cv2.putText(ann, f'L={left_cx:.0f}',
                        (int(left_cx)-20, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
        if right_cx is not None:
            cv2.putText(ann, f'R={right_cx:.0f}',
                        (int(right_cx)-20, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

        cv2.putText(ann, f'err={error:+.2f} V>{v_thr}',
                    (6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

        dbg = {
            'v_thr': v_thr, 'blobs': len(blobs),
            'left_cx': left_cx, 'right_cx': right_cx, 'lane_cx': lane_cx,
        }
        self.last_error_valid = True
        return error, total_px, dbg, ann


# ════════════════════════════════════════════════════════════════════════════
#  TRAFFIC SIGN DETECTOR  (background thread)
# ════════════════════════════════════════════════════════════════════════════

class SignDetector:
    """
    Runs TFLite traffic-sign inference in a background thread.
    Call get_sign() from the main loop to get the latest result.
    """

    MODEL_PATH  = '/opt/vilib/traffic_sign_150_dr0.2.tflite'
    LABELS_PATH = '/opt/vilib/traffic_sign_150_dr0.2_labels.txt'

    def __init__(self, enabled: bool = True):
        self._enabled    = enabled
        self._sign       = 'none'
        self._conf       = 0
        self._lock       = threading.Lock()
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._running    = False
        self._thread     = None
        self._interp     = None
        self._labels     = []
        self._use_opencv = False  # set True when TFLite unavailable

    def start(self):
        if not self._enabled:
            return
        try:
            from tflite_runtime.interpreter import Interpreter
            self._interp = Interpreter(self.MODEL_PATH)
            self._interp.allocate_tensors()
            with open(self.LABELS_PATH) as f:
                self._labels = [l.strip().split(None, 1)[1] if ' ' in l else l.strip()
                                for l in f.readlines()]
            print(' Sign detector: loaded TFLite model OK')
        except Exception as e:
            print(f' Sign detector: TFLite unavailable ({e}) — using OpenCV arrow detection')
            self._use_opencv = True
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def push_frame(self, frame):
        if not self._enabled:
            return
        with self._frame_lock:
            self._frame = frame.copy()

    def get_sign(self):
        with self._lock:
            return self._sign, self._conf

    def _opencv_classify(self, roi):
        """Classify arrow direction from orange/yellow sign blob. No ML needed."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0,  60, 60), (40, 255, 255)),   # red-orange-yellow
            cv2.inRange(hsv, (157, 40, 60), (180, 255, 255)),  # red wrap-around
        )
        k5   = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < SIGN_MIN_AREA or bw < 25 or bh < 25:
                continue
            if min(bw, bh) / max(bw, bh) < 0.35:
                continue

            crop = roi[max(0,y):min(roi.shape[0],y+bh),
                       max(0,x):min(roi.shape[1],x+bw)]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            bg   = int(np.percentile(gray, 60))

            if bg < 120:
                # Sign too dark/blurry to read arrow — use horizontal position:
                # sign on right half of frame means we're approaching a right-turn sign
                sign_cx = x + bw / 2
                if sign_cx / roi.shape[1] > 0.5:
                    return 'right', 72
                else:
                    return 'left', 72

            # Clear sign: find dark arrow pixels vs lighter background
            arrow = gray < max(30, int(bg * 0.7))
            lw = int(np.sum(arrow[:, :gray.shape[1] // 2]))
            rw = int(np.sum(arrow[:, gray.shape[1] // 2:]))
            total = lw + rw

            if total < 15:
                sign_cx = x + bw / 2
                return ('right', 72) if sign_cx / roi.shape[1] > 0.5 else ('left', 72)

            ratio = rw / (total + 1e-6)
            conf  = int(min(90, 50 + abs(ratio - 0.5) * 200))
            lbl   = 'right' if ratio > 0.55 else ('left' if ratio < 0.45 else 'forward')
            return lbl, conf

        return 'none', 0

    def _loop(self):
        if not self._use_opencv:
            in_det  = self._interp.get_input_details()[0]
            out_det = self._interp.get_output_details()[0]
            _, mh, mw, md = in_det['shape']
            idx_in  = in_det['index']
            idx_out = out_det['index']

        while self._running:
            with self._frame_lock:
                frame = self._frame
            if frame is None:
                sleep(0.05)
                continue

            h, w = frame.shape[:2]
            roi  = frame[int(h * SIGN_ROI_TOP):int(h * SIGN_ROI_BOT), :]

            if self._use_opencv:
                best_sign, best_conf = self._opencv_classify(roi)
            else:
                # TFLite path
                hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_r1   = cv2.inRange(hsv, (0,   40, 60), (25,  255, 255))
                mask_r2   = cv2.inRange(hsv, (157, 40, 60), (180, 255, 255))
                mask_blue = cv2.inRange(hsv, (92,  10, 10), (125, 255, 255))
                mask_all  = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_blue)
                k5        = np.ones((5, 5), np.uint8)
                mask_all  = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN,  k5)
                mask_all  = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, k5)
                cnts, _   = cv2.findContours(mask_all, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                best_sign, best_conf = 'none', 0
                for cnt in cnts:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw * bh < SIGN_MIN_AREA or bw < 32 or bh < 32:
                        continue
                    x1 = max(0, x - 8);  y1 = max(0, y - 8)
                    x2 = min(roi.shape[1], x + bw + 8)
                    y2 = min(roi.shape[0], y + bh + 8)
                    crop = roi[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_in = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if md == 1 else crop
                    crop_in = cv2.resize(crop_in, (mw, mh), interpolation=cv2.INTER_LINEAR)
                    crop_in = (crop_in.astype('float32') / 255.0 - 0.5) * 2.0
                    crop_in = np.expand_dims(crop_in.reshape(mh, mw, md), 0)
                    self._interp.set_tensor(idx_in, crop_in)
                    self._interp.invoke()
                    out  = np.squeeze(self._interp.get_tensor(idx_out))
                    conf = int(round(np.max(out) * 100))
                    lbl  = self._labels[int(np.argmax(out))]
                    if conf >= SIGN_CONFIDENCE and conf > best_conf and lbl != 'none':
                        best_conf = conf
                        best_sign = lbl

            with self._lock:
                self._sign = best_sign
                self._conf = best_conf

            sleep(SIGN_INTERVAL)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN FOLLOWER (control loop)
# ════════════════════════════════════════════════════════════════════════════

class LaneFollower:

    # States
    NORMAL      = 'NORMAL'
    STOP_HELD   = 'STOP_HELD'
    TURNING     = 'TURNING'

    def __init__(self, drive_left_lane: bool = False):
        self._detector = LaneDetector(drive_left_lane)

        # PID state
        self._steer    = 0.0
        self._i_err    = 0.0
        self._prev_err = 0.0

        # Gap memory
        self._gap_cnt  = 0
        self._last_err = 0.0

        # State machine
        self._state      = self.NORMAL
        self._state_ts   = monotonic()
        self._turn_dir     = 0      # +1 right, -1 left
        self._sign_acted   = 'none' # last sign we acted on (avoid repeat)
        self._sign_arm_dir = 0      # direction being armed
        self._sign_arm_cnt = 0      # consecutive frames sign has been seen

    # ── Smooth steering ──────────────────────────────────────────────────────
    def _apply_steer(self, target):
        delta         = max(-STEER_RATE, min(STEER_RATE, target - self._steer))
        self._steer   = max(-MAX_STEER, min(MAX_STEER, self._steer + delta))
        return self._steer

    def _speed_for_dist(self, dist, base_speed):
        if 0 < dist < STOP_DIST:
            return 0
        if 0 < dist < SLOW_DIST:
            return min(base_speed, SPEED_CREEP)
        return base_speed

    # ── Grayscale guard ──────────────────────────────────────────────────────
    @staticmethod
    def _gs_alert(gs):
        """
        Returns +1 if right boundary triggered (steer left),
                -1 if left boundary (centre dashes) triggered (steer right),
                 0 if clear.
        """
        if gs[2] > GS_WHITE_THRESHOLD:   # right sensor on white
            return +1
        if gs[0] > GS_WHITE_THRESHOLD:   # left sensor on white (centre dash)
            return -1
        return 0

    # ── PID on lane error ────────────────────────────────────────────────────
    def _pid(self, error):
        self._i_err    = max(-2.0, min(2.0, self._i_err + error))
        d_err          = error - self._prev_err
        self._prev_err = error
        return KP * error + KI * self._i_err + KD * d_err

    # ── Main decision ────────────────────────────────────────────────────────
    def decide(self, frame, gs, dist, sign, sign_conf):
        now = monotonic()

        # ── Obstacle stop ────────────────────────────────────────────────
        if 0 < dist < STOP_DIST and self._state == self.NORMAL:
            return 0.0, 0, 'OBSTACLE stop', frame

        # ── State: executing a turn ──────────────────────────────────────
        if self._state == self.TURNING:
            elapsed = now - self._state_ts
            if elapsed < TURN_HOLD_SEC:
                target = MAX_STEER * self._turn_dir
                steer  = self._apply_steer(target)
                speed  = self._speed_for_dist(dist, SPEED_TURN)
                return steer, speed, f'TURN {"R" if self._turn_dir>0 else "L"} {elapsed:.1f}s', frame
            else:
                self._state    = self.NORMAL
                self._i_err    = 0.0
                self._sign_acted = sign   # mark so we don't retrigger same sign

        # ── State: stopped at stop sign ──────────────────────────────────
        if self._state == self.STOP_HELD:
            elapsed = now - self._state_ts
            if elapsed < STOP_HOLD_SEC:
                return 0.0, 0, f'STOP sign {elapsed:.1f}s', frame
            else:
                self._state      = self.NORMAL
                self._sign_acted = sign

        # ── Sign arming — accumulate frames; turn fires only at lane break ──
        # Only arm when lane is visible (error not None) to avoid false triggers
        # when robot is off-track. We run detection first, then update arm state.
        _lane_visible = self._detector.last_error_valid
        if self._state == self.NORMAL:
            if sign == 'stop' and sign != self._sign_acted:
                self._state    = self.STOP_HELD
                self._state_ts = now
                return 0.0, 0, f'SIGN stop ({sign_conf}%)', frame
            elif sign in ('left', 'right') and sign != self._sign_acted and _lane_visible:
                direction = +1 if sign == 'right' else -1
                if direction == self._sign_arm_dir:
                    self._sign_arm_cnt += 1
                else:
                    self._sign_arm_dir = direction
                    self._sign_arm_cnt = 1
            else:
                self._sign_arm_cnt = 0
                self._sign_arm_dir = 0

        # ── Grayscale fast guard ─────────────────────────────────────────
        gs_alert = self._gs_alert(gs)
        if gs_alert != 0:
            # Override: steer away from boundary immediately
            gs_steer = self._apply_steer(-gs_alert * MAX_STEER * 0.75)
            self._last_err = -gs_alert * 0.5  # bias PID toward recovery
            speed    = self._speed_for_dist(dist, SPEED_CORRECT)
            return gs_steer, speed, f'GS GUARD {"right→left" if gs_alert>0 else "left→right"}', frame

        # ── Camera lane detection ────────────────────────────────────────
        error, total_px, dbg, ann = self._detector.detect(frame)

        if error is None:
            # No lines in view — gap between dashes or car off track
            self._gap_cnt += 1
            self._last_err *= GAP_HOLD_DECAY
            self._i_err = 0.0

            # Armed turn fires when the lane actually breaks
            if (self._sign_arm_cnt >= SIGN_ARM_FRAMES
                    and self._sign_arm_dir != 0
                    and self._gap_cnt >= 2):
                self._state        = self.TURNING
                self._state_ts     = now
                self._turn_dir     = self._sign_arm_dir
                self._i_err        = 0.0
                self._sign_arm_cnt = 0
                self._sign_arm_dir = 0
                self._sign_acted   = sign
                steer = self._apply_steer(MAX_STEER * self._turn_dir)
                speed = self._speed_for_dist(dist, SPEED_TURN)
                return steer, speed, f'TURN {"R" if self._turn_dir>0 else "L"} (sign+gap)', ann

            if self._gap_cnt <= GAP_MAX_FRAMES:
                # Hold last steer
                speed = self._speed_for_dist(dist, SPEED_CORRECT)
                return self._steer, speed, f'GAP hold ({self._gap_cnt})', ann

            # Beyond gap tolerance — creep with last bias
            raw   = self._pid(self._last_err)
            steer = self._apply_steer(raw)
            speed = self._speed_for_dist(dist, SPEED_CREEP)
            return steer, speed, f'GAP seek ({self._gap_cnt})', ann

        # Line found — reset gap counter
        self._gap_cnt  = 0
        self._last_err = error

        # ── PID steering ─────────────────────────────────────────────────
        raw   = self._pid(error)
        steer = self._apply_steer(raw)

        # ── Speed from error magnitude ────────────────────────────────────
        if abs(error) < 0.20:
            speed = self._speed_for_dist(dist, SPEED_CRUISE)
        elif abs(error) < 0.50:
            speed = self._speed_for_dist(dist, SPEED_CORRECT)
        else:
            speed = self._speed_for_dist(dist, SPEED_CREEP)

        lx = f'{dbg["left_cx"]:.0f}'  if dbg.get('left_cx')  is not None else '–'
        rx = f'{dbg["right_cx"]:.0f}' if dbg.get('right_cx') is not None else '–'
        label = (f'LANE err={error:+.2f} '
                 f'L={lx:>5} R={rx:>5} '
                 f'steer={steer:+.1f}° V>{dbg["v_thr"]}')
        return steer, speed, label, ann


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    global CAM_TILT

    ap = argparse.ArgumentParser()
    ap.add_argument('--left-lane',   action='store_true')
    ap.add_argument('--cam-tilt',    type=float, default=CAM_TILT)
    ap.add_argument('--no-signs',    action='store_true')
    ap.add_argument('--save-frames', action='store_true')
    args = ap.parse_args()

    CAM_TILT = args.cam_tilt

    print(f"\n{'═'*62}")
    print(' Picar-X Lane Follower')
    print(f'{'═'*62}')
    print(f' Lane: {"LEFT" if args.left_lane else "RIGHT"}  '
          f'Tilt: {CAM_TILT}°  Signs: {"off" if args.no_signs else "on"}')
    print(f' KP={KP} KI={KI} KD={KD}  '
          f'MAX_STEER={MAX_STEER}° RATE={STEER_RATE}°/frame')
    print(f' CRUISE={SPEED_CRUISE}%  CORRECT={SPEED_CORRECT}%  CREEP={SPEED_CREEP}%')
    print(f'{'═'*62}\n')

    if args.save_frames:
        os.makedirs('/tmp/lf_debug', exist_ok=True)

    # Camera
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={'size': (CAM_W, CAM_H), 'format': 'BGR888'},
        controls={'FrameRate': CAM_FPS},
    )
    cam.configure(cfg)
    cam.start()
    sleep(1.5)
    px.set_cam_pan_angle(CAM_PAN)
    px.set_cam_tilt_angle(CAM_TILT)
    sleep(0.3)
    print(f' Camera ready: {CAM_W}×{CAM_H} tilt={CAM_TILT}°')

    # Sign detector
    sign_det = SignDetector(enabled=not args.no_signs)
    sign_det.start()

    # Follower
    follower = LaneFollower(drive_left_lane=args.left_lane)
    px.set_dir_servo_angle(0)

    print('\n Starting — Ctrl+C to stop.\n')
    header = f" {'State':54s} {'Steer':>6} {'Spd':>4} {'Dist':>5} {'GS':>15} {'Sign':>8}"
    print(header)
    print(' ' + '─' * (len(header) - 1))

    cycle    = 0
    save_cyc = 0
    stopped  = False
    period   = 1.0 / LOOP_HZ

    try:
        while True:
            t0    = monotonic()
            frame = cam.capture_array()
            dist  = px.get_distance()
            gs    = px.get_grayscale_data()
            sign, sign_conf = sign_det.get_sign()

            sign_det.push_frame(frame)

            steer, speed, label, ann = follower.decide(frame, gs, dist, sign, sign_conf)

            if sign != 'none':
                cv2.putText(ann, f'SIGN:{sign}({sign_conf}%)', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            px.set_dir_servo_angle(steer)
            if speed == 0:
                if not stopped:
                    px.stop(); stopped = True
            elif speed < 0:
                px.backward(abs(speed)); stopped = False
            else:
                px.forward(speed); stopped = False

            if cycle % PRINT_EVERY == 0:
                dist_s = f'{dist:.0f}cm' if dist > 0 else '  ---'
                gs_s   = f'[{gs[0]:4.0f},{gs[1]:4.0f},{gs[2]:4.0f}]'
                sign_s = f'{sign}({sign_conf}%)' if sign != 'none' else '    -'
                print(f' {label:54s} {steer:+5.1f}° {speed:3d}% {dist_s:>5} {gs_s:>15} {sign_s:>8}',
                      flush=True)

            if args.save_frames and cycle % 10 == 0:
                cv2.imwrite(f'/tmp/lf_debug/f{save_cyc:05d}.jpg', ann)
                save_cyc += 1

            cycle += 1
            wait = period - (monotonic() - t0)
            if wait > 0:
                sleep(wait)

    except KeyboardInterrupt:
        print('\n\n Ctrl+C — stopping.')
    finally:
        sign_det.stop()
        cam.stop()
        px.stop()
        px.set_dir_servo_angle(0)
        px.set_cam_tilt_angle(0)
        print(' Done.\n')


if __name__ == '__main__':
    main()

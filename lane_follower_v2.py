#!/usr/bin/env python3
"""
Picar-X Lane Follower  v2
=========================
Architecture:
  - Lane keeping  : adaptive HSV white-mask + PID on lane-centre error
  - Junction det. : all 3 grayscale sensors on white simultaneously
  - Sonar gate    : obstacle within JUNCTION_SONAR_CM confirms real junction
                    (not just a dash gap)
  - Sign direction: OpenAI Vision API queried every AI_POLL_SEC seconds in a
                    background thread; requires AI_ARM_THRESHOLD consecutive
                    matching responses before arming a turn
  - Turn execution: fires when armed direction AND (junction marker OR
                    long camera-gap AND obstacle ahead)

Usage:
  python3 lane_follower_v2.py
  python3 lane_follower_v2.py --left-lane
  python3 lane_follower_v2.py --save-frames
  python3 lane_follower_v2.py --no-signs
  python3 lane_follower_v2.py --cam-tilt -20
"""

import argparse
import base64
import os
import threading
from pathlib import Path
from time import monotonic, sleep

import flask

# Load API key from ~/.env if not already in environment
_env_file = Path.home() / '.env'
if _env_file.exists() and 'OPENAI_API_KEY' not in os.environ:
    for _line in _env_file.read_text().splitlines():
        if _line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY'] = _line.split('=', 1)[1].strip()
            break

import cv2
import numpy as np
from picamera2 import Picamera2
from picarx import Picarx

# ── Hardware ─────────────────────────────────────────────────────────────────
px = Picarx()

# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Camera
CAM_W, CAM_H = 640, 480
CAM_FPS      = 30
CAM_TILT     = -15.0
CAM_PAN      =   0.0

# White detection (HSV)
WHITE_V_MIN  = 175
WHITE_S_MAX  =  60
ADAPTIVE_K   =  0.20

# Lane ROI (fraction of frame height) — tuned for CAM_TILT = -15°
LANE_ROI_TOP = 0.30
LANE_ROI_BOT = 0.82

# Sign ROI — upper portion where signs appear ahead of car
SIGN_ROI_TOP = 0.00
SIGN_ROI_BOT = 0.65

# Lane blob filters
MIN_CONTOUR_AREA = 200
MIN_ASPECT_RATIO = 0.8

# Grayscale — raw ADC values (dark mat ≈ 200–400, white tape ≈ 800–2000)
GS_WHITE_THRESHOLD = 600

# PID
KP = 28.0
KI =  0.15
KD =  7.0

# Steering
MAX_STEER  = 28.0   # servo hard limit (degrees)
STEER_RATE =  7.0   # max change per frame

# Speeds (%)
SPEED_CRUISE  = 12
SPEED_CORRECT = 10
SPEED_CREEP   =  7
SPEED_TURN    = 11

# Obstacle / sonar
STOP_DIST          = 15   # cm — full stop
SLOW_DIST          = 40   # cm — reduce speed
JUNCTION_SONAR_CM  = 40   # cm — obstacle this close at a gap = junction wall ahead

# Gap handling (camera sees no lines)
GAP_MAX_FRAMES  = 12     # hold last steer this many frames
GAP_HOLD_DECAY  =  0.97  # per-frame decay of last error during gap
GAP_JUNCTION_MIN =  5    # camera-gap frames needed before sonar gate is checked

# Turn execution
TURN_HOLD_SEC         = 2.3   # seconds to hold full steer during a turn
JUNCTION_LATCH_SEC    = 6.0   # seconds to remember a junction stripe before giving up
JUNCTION_DEFAULT_DIR  = 'right'  # fallback if AI never arms before latch expires
TURN_COOLDOWN   = 5.0    # seconds after turn before new sign arming is allowed
STOP_HOLD_SEC   = 3.0    # seconds to pause at a stop sign

# Stuck recovery
RECOVERY_SONAR_CM    = 14    # cm — only recover when this close (tighter than STOP_DIST)
RECOVERY_TRIGGER_SEC =  2.0  # seconds stopped before recovery begins
RECOVERY_BACK_SEC    =  0.7  # seconds to reverse before AI recovery call
RECOVERY_TURN_SEC    =  1.5  # seconds to turn during recovery
RECOVERY_MAX_TRIES   =  2    # max attempts per stuck event before giving up

# OpenAI sign detector
AI_POLL_SEC      = 1.0   # seconds between API calls (after ~1s API latency = ~2s cycle)
AI_ARM_THRESHOLD =  1    # responses to arm; stripe is the physical gate so 1 is safe
AI_JPEG_QUALITY  = 92    # JPEG quality for API image upload (lower = faster)

# Loop
LOOP_HZ     = 25
PRINT_EVERY =  8


# ═════════════════════════════════════════════════════════════════════════════
#  WEB STREAM  (MJPEG on port 8080 — browse to http://<pi-ip>:8080/stream)
# ═════════════════════════════════════════════════════════════════════════════

_stream_frame     = None
_stream_lock      = threading.Lock()
_stream_flask_app = flask.Flask(__name__)

@_stream_flask_app.route('/')
def _stream_index():
    return '<html><body><img src="/stream"></body></html>'

@_stream_flask_app.route('/stream')
def _stream_feed():
    def _generate():
        while True:
            with _stream_lock:
                frame = _stream_frame
            if frame is not None:
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
            sleep(0.04)
    return flask.Response(_generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

def _start_web_stream():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    _stream_flask_app.run(host='0.0.0.0', port=8080, threaded=True)

def push_web_frame(frame):
    global _stream_frame
    with _stream_lock:
        _stream_frame = frame


# ═════════════════════════════════════════════════════════════════════════════
#  LANE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Detects the two lane boundaries from a camera frame using white blob analysis.

    Returns a normalised error:
       0.0  = centred in lane
      +1.0  = too far right  → steer left
      -1.0  = too far left   → steer right
    """

    def __init__(self, drive_left_lane: bool = False):
        self._drive_left      = drive_left_lane
        self.last_error_valid = False

    def _white_mask(self, roi_bgr):
        hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
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
        Returns (error, total_px, dbg_dict, annotated_frame).
        error is None when no lines are found.
        """
        h, w = frame.shape[:2]
        y0   = int(h * LANE_ROI_TOP)
        y1   = int(h * LANE_ROI_BOT)
        roi  = frame[y0:y1, :]

        mask, v_thr = self._white_mask(roi)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx  = M['m10'] / M['m00']
            cy  = M['m01'] / M['m00']
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw / max(bh, 1) < MIN_ASPECT_RATIO:
                continue
            blobs.append({'cx': cx, 'cy': cy, 'area': area, 'cnt': cnt})

        ann = frame.copy()
        cv2.rectangle(ann, (0, y0), (w - 1, y1), (0, 200, 200), 1)
        cv2.line(ann,      (w // 2, y0), (w // 2, y1), (255, 0, 0), 1)

        if not blobs:
            cv2.putText(ann, 'NO LINES', (10, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            self.last_error_valid = False
            return None, 0, {'v_thr': v_thr, 'blobs': 0}, ann

        left_blobs  = [b for b in blobs if b['cx'] <  w / 2]
        right_blobs = [b for b in blobs if b['cx'] >= w / 2]

        def weighted_cx(blob_list):
            if not blob_list:
                return None
            total_a = sum(b['area'] for b in blob_list)
            return sum(b['cx'] * b['area'] for b in blob_list) / total_a

        left_cx  = weighted_cx(left_blobs)
        right_cx = weighted_cx(right_blobs)

        if left_cx is not None and right_cx is not None:
            lane_cx = (left_cx + right_cx) / 2.0
        elif left_cx is not None:
            lane_cx = (left_cx + w) / 2.0
        elif right_cx is not None:
            lane_cx = right_cx / 2.0
        else:
            self.last_error_valid = False
            return None, 0, {'v_thr': v_thr, 'blobs': 0}, ann

        if self._drive_left:
            lane_cx = w - lane_cx

        error = (lane_cx - w / 2.0) / (w / 2.0)

        # Annotate
        for b in blobs:
            color = (0, 255, 100) if b['cx'] < w / 2 else (0, 100, 255)
            cnt_abs = b['cnt'] + np.array([0, y0])
            cv2.drawContours(ann, [cnt_abs], -1, color, 1)
            cv2.circle(ann, (int(b['cx']), int(b['cy']) + y0), 5, (0, 0, 255), -1)

        cv2.line(ann, (int(lane_cx), y0), (int(lane_cx), y1), (0, 255, 255), 2)
        if left_cx  is not None:
            cv2.putText(ann, f'L={left_cx:.0f}',  (int(left_cx)  - 20, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
        if right_cx is not None:
            cv2.putText(ann, f'R={right_cx:.0f}', (int(right_cx) - 20, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
        cv2.putText(ann, f'err={error:+.2f} V>{v_thr}',
                    (6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

        total_px = sum(b['area'] for b in blobs)
        dbg = {
            'v_thr': v_thr, 'blobs': len(blobs),
            'left_cx': left_cx, 'right_cx': right_cx, 'lane_cx': lane_cx,
        }
        self.last_error_valid = True
        return error, total_px, dbg, ann


# ═════════════════════════════════════════════════════════════════════════════
#  OPENAI SIGN DETECTOR  (background thread)
# ═════════════════════════════════════════════════════════════════════════════

class OpenAISignDetector:
    """
    Periodically sends a camera frame to GPT-4o Vision and asks which direction
    to turn at the next junction.

    Design:
      - Runs in a daemon thread; never blocks the drive loop.
      - Requires AI_ARM_THRESHOLD consecutive identical responses before arming.
      - Armed direction is cleared by calling reset() after a turn executes.
      - If the API call fails, the result is treated as 'none' (safe default).
    """

    PROMPT = (
        "You are the vision system of a small autonomous robot car driving on a "
        "track. Look at this image from the robot's forward-facing camera. "
        "Is there a traffic sign visible? "
        "If it is a STOP sign, reply: stop. "
        "If it indicates a direction to turn at the next junction, reply: left, right, or forward. "
        "If no sign is visible, reply: none. "
        "Reply with exactly ONE word. Do not explain. Do not punctuate."
    )

    RECOVERY_PROMPT = (
        "A small robot car has stopped against an obstacle on a road track and has "
        "reversed slightly to get clearance. Looking at this camera image, should "
        "the robot turn LEFT or RIGHT to get back onto the road lane? "
        "Reply with exactly ONE word — left or right. Do not explain. Do not punctuate."
    )

    def __init__(self, enabled: bool = True):
        self._enabled    = enabled
        self._lock       = threading.Lock()
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._running    = False
        self._thread     = None
        self._client     = None      # openai.OpenAI() created once in start()

        # Arming state (protected by _lock)
        self._pending     = 'none'   # last API response
        self._arm_cnt     = 0        # consecutive matching responses
        self._armed_dir   = 'none'   # confirmed direction (ready to fire)

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self):
        if not self._enabled:
            return
        import openai
        self._client  = openai.OpenAI()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(' OpenAI sign detector: started (gpt-4o-mini, polling every ~2s)')

    def stop(self):
        self._running = False

    def push_frame(self, frame):
        """Call every drive loop iteration with the latest camera frame."""
        if not self._enabled:
            return
        with self._frame_lock:
            self._frame = frame.copy()

    def get_armed_direction(self) -> str:
        """Returns the armed turn direction ('left'/'right'/'forward') or 'none'."""
        with self._lock:
            return self._armed_dir

    def get_status(self) -> tuple[str, int]:
        """Returns (pending_response, arm_count) for debug display."""
        with self._lock:
            return self._pending, self._arm_cnt

    def reset(self):
        """Call after a turn executes to clear armed state."""
        with self._lock:
            self._pending   = 'none'
            self._arm_cnt   = 0
            self._armed_dir = 'none'

    def call_recovery(self, frame) -> str:
        """Synchronous blocking call — asks AI which way to turn after backing away from obstacle."""
        if not self._enabled or self._client is None:
            return 'right'
        b64 = self._frame_to_b64(frame)
        if not b64:
            return 'right'
        try:
            t0 = monotonic()
            response = self._client.chat.completions.create(
                model      = 'gpt-4o-mini',
                max_tokens = 5,
                timeout    = 8.0,
                messages   = [{
                    'role': 'user',
                    'content': [
                        {'type': 'text',      'text': self.RECOVERY_PROMPT},
                        {'type': 'image_url', 'image_url': {
                            'url':    f'data:image/jpeg;base64,{b64}',
                            'detail': 'low',
                        }},
                    ],
                }],
            )
            word    = response.choices[0].message.content.strip().lower()
            elapsed = monotonic() - t0
            result  = word if word in ('left', 'right') else 'right'
            print(f' [AI-RECOVERY] {result}  ({elapsed:.1f}s)', flush=True)
            return result
        except Exception as e:
            print(f' [AI-RECOVERY] error: {e} — defaulting right', flush=True)
            return 'right'

    # ── Internal ─────────────────────────────────────────────────────────────

    def _frame_to_b64(self, frame) -> str:
        """JPEG-encode frame and return as base64 string for the API."""
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, AI_JPEG_QUALITY])
        if not ok:
            return ''
        return base64.b64encode(buf.tobytes()).decode('utf-8')

    def _call_openai(self, frame) -> str:
        """Send frame to OpenAI Vision and return 'left', 'right', 'forward', or 'none'."""
        from time import monotonic
        h, w = frame.shape[:2]
        roi  = frame[int(h * SIGN_ROI_TOP):int(h * SIGN_ROI_BOT), :]
        b64  = self._frame_to_b64(roi)
        if not b64:
            return 'none'

        t0       = monotonic()
        response = self._client.chat.completions.create(
            model      = 'gpt-4o-mini',
            max_tokens = 5,
            timeout    = 8.0,
            messages   = [{
                'role': 'user',
                'content': [
                    {'type': 'text',      'text': self.PROMPT},
                    {'type': 'image_url', 'image_url': {
                        'url':    f'data:image/jpeg;base64,{b64}',
                        'detail': 'low',
                    }},
                ],
            }],
        )
        word    = response.choices[0].message.content.strip().lower()
        elapsed = monotonic() - t0
        result  = word if word in ('left', 'right', 'forward', 'stop') else 'none'
        print(f' [AI] {result:>8}  ({elapsed:.1f}s)', flush=True)
        return result

    def _loop(self):
        while self._running:
            with self._frame_lock:
                frame = self._frame

            if frame is None:
                sleep(0.1)
                continue

            try:
                response = self._call_openai(frame)
            except Exception as e:
                print(f' [OpenAI] error: {e}')
                response = 'none'

            with self._lock:
                if response == 'none':
                    # No sign visible — decay arm count gradually so a single
                    # missed frame doesn't wipe a near-armed state
                    self._arm_cnt = max(0, self._arm_cnt - 1)
                    if self._arm_cnt == 0:
                        self._pending = 'none'
                elif response == 'stop':
                    # Stop sign arms immediately regardless of threshold
                    self._pending   = 'stop'
                    self._arm_cnt   = AI_ARM_THRESHOLD
                    self._armed_dir = 'stop'
                elif response == self._pending:
                    self._arm_cnt += 1
                    if self._arm_cnt >= AI_ARM_THRESHOLD:
                        self._armed_dir = response
                else:
                    # Direction changed — restart count
                    self._pending = response
                    self._arm_cnt = 1
                    if self._arm_cnt >= AI_ARM_THRESHOLD:
                        self._armed_dir = response

            sleep(AI_POLL_SEC)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN FOLLOWER (state machine + control loop)
# ═════════════════════════════════════════════════════════════════════════════

class LaneFollower:

    NORMAL     = 'NORMAL'
    TURNING    = 'TURNING'
    STOP_HELD  = 'STOP_HELD'
    RECOVERING = 'RECOVERING'

    def __init__(self, drive_left_lane: bool = False):
        self._detector = LaneDetector(drive_left_lane)

        # PID
        self._steer    = 0.0
        self._i_err    = 0.0
        self._prev_err = 0.0

        # Gap tracking
        self._gap_cnt  = 0
        self._last_err = 0.0

        # State machine
        self._state      = self.NORMAL
        self._state_ts   = monotonic()
        self._turn_dir   = 0      # +1 right, -1 left
        self._last_turn_end = 0.0 # monotonic timestamp when last turn finished

        # Junction latch — persists after stripe until AI responds or timeout
        self._junction_seen = False
        self._junction_ts   = 0.0

        # Stuck recovery
        self._obstacle_stop_ts  = 0.0   # when current obstacle stop started (0 = not stopped)
        self._recovery_attempts = 0     # attempts for current stuck event
        self._recovery_dir      = 0     # +1 right, -1 left
        self._recovery_phase    = None  # 'backup' | 'turn'
        self._recovery_ts       = 0.0

        # Stop sign exit flag
        self.should_exit = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _apply_steer(self, target: float) -> float:
        delta       = max(-STEER_RATE, min(STEER_RATE, target - self._steer))
        self._steer = max(-MAX_STEER,  min(MAX_STEER,  self._steer + delta))
        return self._steer

    def _speed_for_dist(self, dist: float, base: int) -> int:
        if 0 < dist < STOP_DIST:
            return 0
        if 0 < dist < SLOW_DIST:
            return min(base, SPEED_CREEP)
        return base

    def _pid(self, error: float) -> float:
        self._i_err    = max(-2.0, min(2.0, self._i_err + error))
        d_err          = error - self._prev_err
        self._prev_err = error
        return KP * error + KI * self._i_err + KD * d_err

    @staticmethod
    def _junction_gs(gs) -> bool:
        """All three grayscale sensors on white = perpendicular junction marker."""
        return all(v > GS_WHITE_THRESHOLD for v in gs)

    @staticmethod
    def _boundary_alert(gs) -> int:
        """
        Single-sensor boundary alerts for in-lane corrections.
        Returns +1 (steer left), -1 (steer right), or 0 (clear).
        Only checked when NOT at a junction (not all-three-on-white).
        """
        if gs[2] > GS_WHITE_THRESHOLD:   # right sensor → steer left
            return +1
        if gs[0] > GS_WHITE_THRESHOLD:   # left sensor → steer right
            return -1
        return 0

    @staticmethod
    def _obstacle_ahead(dist: float) -> bool:
        """Sonar confirms a wall/boundary closer than JUNCTION_SONAR_CM."""
        return 0 < dist < JUNCTION_SONAR_CM

    def _in_cooldown(self) -> bool:
        return (monotonic() - self._last_turn_end) < TURN_COOLDOWN

    # ── Main decision ─────────────────────────────────────────────────────────

    def decide(self, frame, gs, dist, armed_dir: str, sign_det: OpenAISignDetector):
        """
        Returns (steer_deg, speed_pct, label_str, annotated_frame).

        armed_dir : the direction armed by OpenAISignDetector ('left'/'right'/
                    'forward'/'none')
        sign_det  : reference to the detector so we can call reset() on it
        """
        now = monotonic()

        # ── Hard stop (obstacle very close) — suppressed at junction ─────
        if 0 < dist < STOP_DIST and self._state == self.NORMAL and not self._junction_seen:
            if self._obstacle_stop_ts == 0.0:
                self._obstacle_stop_ts = now
            stopped_for = now - self._obstacle_stop_ts
            if (dist <= RECOVERY_SONAR_CM
                    and stopped_for >= RECOVERY_TRIGGER_SEC
                    and self._recovery_attempts < RECOVERY_MAX_TRIES):
                self._recovery_attempts += 1
                self._recovery_phase = 'backup'
                self._recovery_ts    = now
                self._state          = self.RECOVERING
                self._junction_seen  = False
                print(f' [RECOVERY] attempt {self._recovery_attempts}/{RECOVERY_MAX_TRIES} — backing up', flush=True)
                return 0.0, -SPEED_CREEP, 'RECOVERY start', frame
            tag = 'RECOVERY EXHAUSTED' if self._recovery_attempts >= RECOVERY_MAX_TRIES else f'stuck {stopped_for:.1f}s'
            return 0.0, 0, f'OBSTACLE {dist:.0f}cm  {tag}', frame
        elif self._state == self.NORMAL and self._obstacle_stop_ts != 0.0:
            self._obstacle_stop_ts  = 0.0
            self._recovery_attempts = 0

        # ── State: executing a turn ───────────────────────────────────────
        if self._state == self.TURNING:
            elapsed = now - self._state_ts
            if elapsed < TURN_HOLD_SEC:
                target = MAX_STEER * self._turn_dir
                steer  = self._apply_steer(target)
                speed  = self._speed_for_dist(dist, SPEED_TURN)
                return steer, speed, f'TURN {"R" if self._turn_dir>0 else "L"} {elapsed:.1f}s', frame
            else:
                self._state         = self.NORMAL
                self._i_err         = 0.0
                self._last_turn_end = now
                self._junction_seen = False
                sign_det.reset()

        # ── State: stopped at stop sign ───────────────────────────────────
        if self._state == self.STOP_HELD:
            elapsed = now - self._state_ts
            if elapsed < STOP_HOLD_SEC:
                return 0.0, 0, f'STOP {elapsed:.1f}s', frame
            else:
                self._state         = self.NORMAL
                self._last_turn_end = now
                sign_det.reset()

        # ── State: obstacle stuck recovery ───────────────────────────────
        if self._state == self.RECOVERING:
            elapsed = now - self._recovery_ts
            if self._recovery_phase == 'backup':
                if elapsed < RECOVERY_BACK_SEC:
                    return 0.0, -SPEED_CREEP, f'RECOVERY backup {elapsed:.1f}s', frame
                # Backup done — pick direction from last known lane error (no AI call)
                px.stop()
                sleep(0.2)
                if self._recovery_attempts == 1:
                    # Use the same direction as the last junction turn so the car
                    # re-enters the curve it was navigating. _turn_dir: +1=right, -1=left.
                    rec_dir = 'right' if self._turn_dir >= 0 else 'left'
                    print(f' [RECOVERY] attempt 1: last_turn={self._turn_dir:+d} → {rec_dir}', flush=True)
                else:
                    rec_dir = 'left' if self._recovery_dir > 0 else 'right'
                    print(f' [RECOVERY] attempt {self._recovery_attempts}: flipping to {rec_dir}', flush=True)
                self._recovery_dir   = +1 if rec_dir == 'right' else -1
                self._recovery_phase = 'turn'
                self._recovery_ts    = now
            if self._recovery_phase == 'turn':
                elapsed = now - self._recovery_ts
                if elapsed < RECOVERY_TURN_SEC:
                    steer = self._apply_steer(MAX_STEER * self._recovery_dir)
                    return steer, SPEED_TURN, f'RECOVERY turn {"R" if self._recovery_dir>0 else "L"} {elapsed:.1f}s', frame
                # Turn done — resume normal driving
                self._state          = self.NORMAL
                self._recovery_phase = None
                self._i_err          = 0.0
                self._obstacle_stop_ts = 0.0
                print(' [RECOVERY] done — resuming', flush=True)

        # ── Stop sign: stop car and signal program exit ───────────────────
        if armed_dir == 'stop' and self._state == self.NORMAL:
            print(' [STOP SIGN] detected — stopping and exiting.', flush=True)
            self.should_exit = True
            return 0.0, 0, 'STOP SIGN — exiting', frame

        # ── Junction latch: fire as soon as AI arms after stripe ─────────
        # The stripe sets _junction_seen; AI response may arrive seconds later.
        # This check fires the turn the moment both conditions are satisfied.
        if self._junction_seen and not self._in_cooldown():
            if armed_dir != 'none':
                self._junction_seen = False
                return self._fire_turn(armed_dir, dist, 'LATCH+AI', frame, sign_det, now)
            if (now - self._junction_ts) > JUNCTION_LATCH_SEC:
                self._junction_seen = False
                print(f' [JUNCTION] latch expired — no AI response, stopping.', flush=True)
                # default fallback disabled — testing pure AI response
                # return self._fire_turn(JUNCTION_DEFAULT_DIR, dist, 'latch-timeout', frame, sign_det, now)
            # Still waiting — hold position, print every cycle
            wait_s = now - self._junction_ts
            print(f' [JUNCTION] waiting for AI...  {wait_s:.1f}s / {JUNCTION_LATCH_SEC:.0f}s  armed={armed_dir}', flush=True)
            return 0.0, 0, f'JUNCTION wait AI {wait_s:.1f}s', frame

        # ── Junction stripe detection (GS all-white) ──────────────────────
        at_junction_gs = self._junction_gs(gs)
        if at_junction_gs and not self._in_cooldown():
            if armed_dir != 'none':
                # Already armed — fire immediately
                return self._fire_turn(armed_dir, dist, 'GS+armed', frame, sign_det, now)
            # Not armed yet — latch and slow down to give AI time to respond
            if not self._junction_seen:
                self._junction_seen = True
                self._junction_ts   = now
                print(f' [JUNCTION] stripe latched  gs={gs}  armed={armed_dir}', flush=True)
                os.makedirs('/tmp/lf_debug', exist_ok=True)
                cv2.imwrite('/tmp/lf_debug/junction_latch.jpg', frame)
                h, w2 = frame.shape[:2]
                sign_roi = frame[int(h * SIGN_ROI_TOP):int(h * SIGN_ROI_BOT), :]
                cv2.imwrite('/tmp/lf_debug/junction_sign_roi.jpg', sign_roi)
            speed = self._speed_for_dist(dist, SPEED_CREEP)
            return self._apply_steer(0.0), speed, 'JUNCTION latched (no arm)', frame

        # ── In-lane boundary guard (single sensor) ────────────────────────
        alert = self._boundary_alert(gs)
        if alert != 0:
            gs_steer       = self._apply_steer(-alert * MAX_STEER * 0.75)
            self._last_err = -alert * 0.5
            speed          = self._speed_for_dist(dist, SPEED_CORRECT)
            label          = f'GS GUARD {"R→L" if alert > 0 else "L→R"}'
            return gs_steer, speed, label, frame

        # ── Camera lane detection ─────────────────────────────────────────
        error, total_px, dbg, ann = self._detector.detect(frame)

        if error is None:
            self._gap_cnt  += 1
            self._last_err *= GAP_HOLD_DECAY
            self._i_err     = 0.0

            # Fallback junction detection: long camera-gap alone
            # (fires only if a direction is already armed by AI)
            if (self._gap_cnt >= GAP_JUNCTION_MIN
                    and armed_dir != 'none'
                    and not self._in_cooldown()):
                return self._fire_turn(armed_dir, dist, 'cam-gap', ann, sign_det, now)

            if self._gap_cnt <= GAP_MAX_FRAMES:
                speed = self._speed_for_dist(dist, SPEED_CORRECT)
                return self._steer, speed, f'GAP hold ({self._gap_cnt})', ann

            raw   = self._pid(self._last_err)
            steer = self._apply_steer(raw)
            speed = self._speed_for_dist(dist, SPEED_CREEP)
            return steer, speed, f'GAP seek ({self._gap_cnt})', ann

        # Line found — reset gap counter
        self._gap_cnt  = 0
        self._last_err = error

        # ── PID steering ──────────────────────────────────────────────────
        raw   = self._pid(error)
        steer = self._apply_steer(raw)

        if abs(error) < 0.20:
            speed = self._speed_for_dist(dist, SPEED_CRUISE)
        elif abs(error) < 0.30:
            speed = self._speed_for_dist(dist, SPEED_CORRECT)
        else:
            speed = self._speed_for_dist(dist, SPEED_CREEP)

        lx = f'{dbg["left_cx"]:.0f}'  if dbg.get('left_cx')  is not None else '–'
        rx = f'{dbg["right_cx"]:.0f}' if dbg.get('right_cx') is not None else '–'
        label = (f'LANE err={error:+.2f} L={lx:>5} R={rx:>5} '
                 f'steer={steer:+.1f}° V>{dbg["v_thr"]}')
        return steer, speed, label, ann

    def _fire_turn(self, direction: str, dist: float, trigger: str,
                   frame, sign_det: OpenAISignDetector, now: float):
        """Commit to a turn and return the first frame's drive command."""
        if direction == 'forward':
            # 'forward' means go straight through — no turn needed
            self._last_turn_end = now
            sign_det.reset()
            return 0.0, self._speed_for_dist(dist, SPEED_CORRECT), f'JUNCTION fwd ({trigger})', frame

        self._state    = self.TURNING
        self._state_ts = now
        self._turn_dir = +1 if direction == 'right' else -1
        self._i_err    = 0.0
        steer = self._apply_steer(MAX_STEER * self._turn_dir)
        speed = self._speed_for_dist(dist, SPEED_TURN)
        label = f'TURN {"R" if self._turn_dir > 0 else "L"} 0.0s ({trigger})'
        return steer, speed, label, frame


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global CAM_TILT

    ap = argparse.ArgumentParser(description='Picar-X Lane Follower v2')
    ap.add_argument('--left-lane',   action='store_true',
                    help='Drive in left lane instead of right')
    ap.add_argument('--cam-tilt',    type=float, default=CAM_TILT,
                    help=f'Camera tilt degrees, negative = down (default {CAM_TILT})')
    ap.add_argument('--no-signs',    action='store_true',
                    help='Disable OpenAI sign detection')
    ap.add_argument('--save-frames', action='store_true',
                    help='Save annotated debug frames to /tmp/lf_debug/')
    args = ap.parse_args()

    CAM_TILT = args.cam_tilt

    print(f"\n{'═'*64}")
    print(' Picar-X Lane Follower  v2')
    print(f'{"═"*64}')
    print(f' Lane  : {"LEFT" if args.left_lane else "RIGHT"}')
    print(f' Tilt  : {CAM_TILT}°')
    print(f' Signs : {"DISABLED" if args.no_signs else "OpenAI Vision (async)"}')
    print(f' PID   : KP={KP} KI={KI} KD={KD}')
    print(f' Speeds: cruise={SPEED_CRUISE}% correct={SPEED_CORRECT}% creep={SPEED_CREEP}%')
    print(f'{"═"*64}\n')

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
    print(f' Camera ready: {CAM_W}×{CAM_H} @ {CAM_FPS}fps  tilt={CAM_TILT}°')

    # Web stream
    _wt = threading.Thread(target=_start_web_stream, daemon=True)
    _wt.start()
    print(f' Web stream : http://0.0.0.0:8080/stream  (open in browser)')

    # Sign detector
    sign_det = OpenAISignDetector(enabled=not args.no_signs)
    sign_det.start()

    # Follower
    follower = LaneFollower(drive_left_lane=args.left_lane)
    px.set_dir_servo_angle(0)

    print('\n Starting — Ctrl+C to stop.\n')
    hdr = f" {'State':56s} {'Steer':>6} {'Spd':>4} {'Dist':>5} {'GS':>15} {'AI':>12}"
    print(hdr)
    print(' ' + '─' * (len(hdr) - 1))

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

            sign_det.push_frame(frame)
            armed_dir = sign_det.get_armed_direction()

            steer, speed, label, ann = follower.decide(
                frame, gs, dist, armed_dir, sign_det
            )

            # Overlay AI status on frame
            ai_pending, ai_cnt = sign_det.get_status()
            if armed_dir != 'none':
                cv2.putText(ann, f'AI ARMED:{armed_dir}', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif ai_pending != 'none':
                cv2.putText(ann, f'AI:{ai_pending}({ai_cnt}/{AI_ARM_THRESHOLD})', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            push_web_frame(ann)

            px.set_dir_servo_angle(steer)
            if speed == 0:
                if not stopped:
                    px.stop(); stopped = True
            elif speed < 0:
                px.backward(abs(speed)); stopped = False
            else:
                px.forward(speed); stopped = False

            if follower.should_exit:
                print('\n [STOP SIGN] clean exit.', flush=True)
                break

            if cycle % PRINT_EVERY == 0:
                dist_s = f'{dist:.0f}cm' if dist > 0 else '   ---'
                gs_s   = f'[{gs[0]:4.0f},{gs[1]:4.0f},{gs[2]:4.0f}]'
                ai_s   = f'{armed_dir}' if armed_dir != 'none' else (
                          f'{ai_pending}({ai_cnt})' if ai_pending != 'none' else '-')
                print(f' {label:56s} {steer:+5.1f}° {speed:3d}% {dist_s:>5}'
                      f' {gs_s:>15} {ai_s:>12}', flush=True)

            if args.save_frames:
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

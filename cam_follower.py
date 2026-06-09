#!/usr/bin/env python3
"""
Picar-X Camera-Based White Dashed-Line Follower
------------------------------------------------
Tilts the camera down at the road and uses OpenCV to detect the
white dashed centerline. PID steering on the horizontal offset.

Usage:
  python3 cam_follower.py                  # drive
  python3 cam_follower.py --calibrate      # tune colour thresholds live
  python3 cam_follower.py --cam-tilt -25   # adjust camera angle
  python3 cam_follower.py --save-frames    # save debug frames to /tmp/cam_debug/
"""

from picarx import Picarx
from picamera2 import Picamera2
from time import sleep, monotonic
import cv2
import numpy as np
import os

# ════════════════════════════════════════════════════════════════
#  TUNING
# ════════════════════════════════════════════════════════════════

# ── Camera ───────────────────────────────────────────────────────
CAM_W        = 320
CAM_H        = 240
CAM_FPS      = 30
CAM_TILT     = -30.0   # degrees, negative = look down at road
CAM_PAN      =   0.0   # centred

# ── White detection (HSV colour space) ───────────────────────────
# H=0-180 (any hue), S=low (unsaturated = white/grey), V=high (bright)
# Run --calibrate to find the right values for your lighting.
WHITE_V_MIN  = 190     # minimum brightness to count as white
WHITE_S_MAX  =  50     # maximum saturation  (keeps grey/dark out)

# ── Region of interest ───────────────────────────────────────────
# Only look at the bottom portion of the frame (road just ahead).
# 0.0 = top of frame, 1.0 = bottom of frame.
ROI_TOP      = 0.55    # use bottom 45% of frame

# ── Minimum white pixels to count as "line found" ────────────────
MIN_WHITE_PX = 200

# ── PID gains ────────────────────────────────────────────────────
KP           = 28.0    # proportional: offset fraction → steer degrees
KI           =  0.8    # integral: corrects steady drift
KD           =  4.0    # derivative: damps oscillation

# ── Speeds ───────────────────────────────────────────────────────
FOLLOW_SPEED =  11     # normal line-following speed
SEEK_SPEED   =   7     # creep when in gap between dashes

# ── Steering ─────────────────────────────────────────────────────
MAX_STEER    =  28.0   # degrees hard limit
STEER_RATE   =  10.0   # max degrees change per frame (smoothing)

# ── Gap tolerance ────────────────────────────────────────────────
GAP_FRAMES   =   8     # frames without white before slowing to seek

# ── Obstacle ─────────────────────────────────────────────────────
STOP_DIST    =  15     # cm — full stop
SLOW_DIST    =  35     # cm — reduce to SEEK_SPEED

# ── Loop ─────────────────────────────────────────────────────────
LOOP_HZ      =  30
PRINT_EVERY  =  10     # print every N cycles

# ════════════════════════════════════════════════════════════════
#  INIT
# ════════════════════════════════════════════════════════════════

px = Picarx()


# ════════════════════════════════════════════════════════════════
#  FOLLOWER
# ════════════════════════════════════════════════════════════════

class CameraFollower:
    """PID line follower using camera frames."""

    def __init__(self):
        self._steer    =  0.0
        self._i_err    =  0.0
        self._prev_err =  0.0
        self._gap_cnt  =  0
        self._last_off =  0.0   # last known offset (for gap bias)
        self._obs_cnt  =  0
        self._obs_stuck=  0

    def _apply_steer(self, target):
        delta = max(-STEER_RATE, min(STEER_RATE, target - self._steer))
        self._steer = max(-MAX_STEER, min(MAX_STEER, self._steer + delta))
        return self._steer

    def _speed_cap(self, dist, base):
        if 0 < dist < SLOW_DIST:
            return min(base, SEEK_SPEED)
        return base

    def detect_line(self, frame):
        """
        Find the white line in the ROI using column-weighted centre of mass.

        Returns:
            offset   : float -1.0 (line full-left) … 0 (centre) … +1.0 (full-right)
            white_px : int   number of white pixels found (0 = no line)
            mask     : the binary mask (for debug saving)
        """
        h, w  = frame.shape[:2]
        roi   = frame[int(h * ROI_TOP):, :]
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(
            hsv,
            np.array([0,           0, WHITE_V_MIN], np.uint8),
            np.array([180, WHITE_S_MAX,        255], np.uint8),
        )
        # Morphological open: remove tiny specks
        kernel   = np.ones((3, 3), np.uint8)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        white_px = int(mask.sum() // 255)
        if white_px < MIN_WHITE_PX:
            return 0.0, 0, mask

        col_sum = mask.sum(axis=0).astype(float)
        total   = col_sum.sum()
        if total == 0:
            return 0.0, 0, mask

        cx     = float(np.dot(np.arange(w), col_sum)) / total
        offset = (cx - w / 2.0) / (w / 2.0)
        return offset, white_px, mask

    def decide(self, frame, dist):
        """
        Returns (steer_deg, speed_pct, label, annotated_frame)
        """
        # ── Obstacle ────────────────────────────────────────────
        if 0 < dist < STOP_DIST:
            self._obs_cnt += 1
        else:
            self._obs_cnt = 0

        if self._obs_cnt >= 3:
            self._obs_stuck += 1
            if self._obs_stuck > 50:
                self._obs_stuck = 0
                steer = self._apply_steer(-MAX_STEER * (self._last_off or 1))
                return steer, -SEEK_SPEED, 'OBSTACLE/reverse_escape', frame
            return 0.0, 0, f'OBSTACLE/stop ({dist:.0f}cm)', frame
        else:
            self._obs_stuck = 0

        offset, white_px, mask = self.detect_line(frame)

        # ── Gap between dashes ───────────────────────────────────
        if white_px < MIN_WHITE_PX:
            self._gap_cnt += 1
            self._i_err    = 0.0   # reset integrator — don't wind up in gaps
            if self._gap_cnt >= GAP_FRAMES:
                target = self._last_off * MAX_STEER * 0.25
                steer  = self._apply_steer(target)
                speed  = self._speed_cap(dist, SEEK_SPEED)
                return steer, speed, f'GAP/seek ({self._gap_cnt}) bias={target:+.1f}°', frame
            return self._steer, self._speed_cap(dist, FOLLOW_SPEED), 'GAP/glitch', frame

        # Line found — reset gap counter
        self._gap_cnt  = 0
        self._last_off = offset

        # ── PID ──────────────────────────────────────────────────
        err            = offset
        self._i_err    = max(-1.5, min(1.5, self._i_err + err))
        d_err          = err - self._prev_err
        self._prev_err = err

        raw_steer = KP * err + KI * self._i_err + KD * d_err
        steer     = self._apply_steer(raw_steer)
        speed     = self._speed_cap(dist, FOLLOW_SPEED if abs(offset) < 0.55 else SEEK_SPEED)

        # ── Annotate frame ───────────────────────────────────────
        h, w  = frame.shape[:2]
        roi_y = int(h * ROI_TOP)
        cx_px = int((offset + 1.0) / 2.0 * w)
        mid_y = roi_y + (h - roi_y) // 2
        cv2.line(frame,  (w//2, roi_y), (w//2, h),     (255, 0,   0), 1)   # blue centre
        cv2.circle(frame, (cx_px, mid_y),           6, (0,   255, 0), -1)  # green dot
        cv2.rectangle(frame, (0, roi_y), (w, h),       (0,   200, 200), 1) # ROI box

        label = f'LINE off={offset:+.2f} px={white_px} steer={steer:+.1f}°'
        return steer, speed, label, frame


# ════════════════════════════════════════════════════════════════
#  CALIBRATION MODE
# ════════════════════════════════════════════════════════════════

def run_calibration(cam, seconds=12):
    """
    Live HSV readout so you can tune WHITE_V_MIN / WHITE_S_MAX.

    Point the robot at the white dash → white_px should be high.
    Point at dark foam or grey carpet → white_px should be near 0.
    """
    print(f"\n{'═'*64}")
    print(" CAMERA CALIBRATION — move robot over each surface")
    print(f"{'═'*64}")
    print(f" Current: WHITE_V_MIN={WHITE_V_MIN}  WHITE_S_MAX={WHITE_S_MAX}")
    print(f" ROI = bottom {int((1-ROI_TOP)*100)}% of frame\n")
    print(f" {'Surface':20s}  HSV-centre              white_px")
    print(f" {'-'*20}  {'-'*22}  {'-'*8}")
    sleep(1.0)
    t_end = monotonic() + seconds
    while monotonic() < t_end:
        frame  = cam.capture_array()
        h, w   = frame.shape[:2]
        roi    = frame[int(h * ROI_TOP):, :]
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask   = cv2.inRange(hsv,
                             np.array([0,           0, WHITE_V_MIN], np.uint8),
                             np.array([180, WHITE_S_MAX,        255], np.uint8))
        wpx    = int(mask.sum() // 255)
        cy, cx = roi.shape[0]//2, roi.shape[1]//2
        hc     = hsv[cy, cx]
        rem    = t_end - monotonic()
        print(f"  [{rem:4.1f}s]  centre HSV=({hc[0]:3d}, {hc[1]:3d}, {hc[2]:3d})"
              f"   white_px={wpx:5d}",
              end='\r', flush=True)
        sleep(0.12)
    print(f"\n\n Done. Adjust WHITE_V_MIN / WHITE_S_MAX if needed.\n{'═'*64}\n")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    global WHITE_V_MIN, WHITE_S_MAX, CAM_TILT

    import argparse
    p = argparse.ArgumentParser(description='Picar-X camera line follower')
    p.add_argument('--calibrate',   action='store_true',
                   help='Show live HSV values to tune thresholds, then exit')
    p.add_argument('--cam-tilt',    type=float, default=CAM_TILT,
                   help=f'Camera tilt angle degrees, negative=down (default {CAM_TILT})')
    p.add_argument('--white-v',     type=int,   default=WHITE_V_MIN,
                   help=f'Min HSV V for white detection (default {WHITE_V_MIN})')
    p.add_argument('--white-s',     type=int,   default=WHITE_S_MAX,
                   help=f'Max HSV S for white detection (default {WHITE_S_MAX})')
    p.add_argument('--save-frames', action='store_true',
                   help='Save annotated frames to /tmp/cam_debug/ every 0.5s')
    args = p.parse_args()

    WHITE_V_MIN = args.white_v
    WHITE_S_MAX = args.white_s
    CAM_TILT    = args.cam_tilt

    print(f"\n{'═'*60}")
    print(" Picar-X Camera Line Follower")
    print(f"{'═'*60}")
    print(f" CAM_TILT={CAM_TILT}°  WHITE_V_MIN={WHITE_V_MIN}  WHITE_S_MAX={WHITE_S_MAX}")
    print(f" FOLLOW={FOLLOW_SPEED}%  SEEK={SEEK_SPEED}%  STOP_DIST={STOP_DIST}cm")
    print(f"{'═'*60}\n")

    # ── Camera init ──────────────────────────────────────────────
    print(" Initialising camera...")
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={'size': (CAM_W, CAM_H), 'format': 'BGR888'},
        controls={'FrameRate': CAM_FPS},
    )
    cam.configure(cfg)
    cam.start()
    sleep(1.5)   # let auto-exposure settle
    print(f" Camera ready: {CAM_W}x{CAM_H} @ {CAM_FPS}fps")

    # ── Aim camera at road ───────────────────────────────────────
    px.set_cam_pan_angle(CAM_PAN)
    px.set_cam_tilt_angle(CAM_TILT)
    sleep(0.5)
    print(f" Camera aimed: tilt={CAM_TILT}°  pan={CAM_PAN}°\n")

    if args.save_frames:
        os.makedirs('/tmp/cam_debug', exist_ok=True)
        print(" Debug frames → /tmp/cam_debug/\n")

    if args.calibrate:
        run_calibration(cam, seconds=12)
        cam.stop()
        px.stop()
        return

    follower = CameraFollower()
    px.set_dir_servo_angle(0)

    print(" Starting drive loop.  Ctrl+C to stop.\n")
    print(f" {'State':54s} {'Steer':>6} {'Spd':>4} {'Dist':>6}")
    print(f" {'-'*54} {'-'*6} {'-'*4} {'-'*6}")

    cycle      = 0
    stopped    = False
    save_cycle = 0
    period     = 1.0 / LOOP_HZ

    try:
        while True:
            t0    = monotonic()
            frame = cam.capture_array()
            dist  = px.get_distance()

            steer, speed, label, ann_frame = follower.decide(frame, dist)

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
                print(f" {label:54s} {steer:+5.0f}° {speed:3d}%  {dist_s:>5}", flush=True)

            if args.save_frames and cycle % 15 == 0:
                cv2.imwrite(f'/tmp/cam_debug/f{save_cycle:05d}.jpg', ann_frame)
                save_cycle += 1

            cycle += 1
            wait = period - (monotonic() - t0)
            if wait > 0:
                sleep(wait)

    except KeyboardInterrupt:
        print("\n\n Ctrl+C — stopping.")
    except Exception as e:
        print(f"\n\n ERROR: {e}")
        raise
    finally:
        cam.stop()
        px.stop()
        px.set_dir_servo_angle(0)
        px.set_cam_tilt_angle(0)
        print(" Camera and motors stopped.\n")


if __name__ == '__main__':
    main()

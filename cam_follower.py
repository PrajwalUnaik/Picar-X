#!/usr/bin/env python3
"""
Picar-X Camera-Based White Dashed-Line Follower  (v2)
------------------------------------------------------
Improvements over v1:
  - Multi-band scan (near / mid / far) gives line angle for look-ahead steering
  - Angle-aware PID: steers toward where the line IS and where it's GOING
  - Confidence-based speed: faster when detection is solid
  - Contour shape filter: rejects non-line blobs (too round/small)
  - Adaptive brightness: threshold relative to frame mean so lighting changes
    don't break detection
  - Tighter ROI: avoids robot body in upper frame
  - Higher resolution: 640×480 for better precision

Usage:
  python3 cam_follower.py                  # drive
  python3 cam_follower.py --calibrate      # tune thresholds live
  python3 cam_follower.py --cam-tilt -35   # try different angle
  python3 cam_follower.py --save-frames    # save debug JPEG every 0.5 s
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
CAM_W        = 640
CAM_H        = 480
CAM_FPS      = 30
CAM_TILT     = -38.0   # steeper → sees past robot body; try -35 to -42
CAM_PAN      =   0.0

# ── White detection (HSV) ────────────────────────────────────────
# These are starting values; run --calibrate to tune for your lighting.
WHITE_V_MIN  = 185     # minimum HSV V (brightness) for white
WHITE_S_MAX  =  55     # maximum HSV S (saturation)  for white
# Adaptive offset: threshold = WHITE_V_MIN + ADAPTIVE_K * (mean_V - 128)
# This raises/lowers the threshold with ambient brightness.
ADAPTIVE_K   =  0.25   # 0 = fixed threshold, 1 = fully adaptive

# ── Region of interest ───────────────────────────────────────────
# Only look at a narrow band near the bottom of the frame.
# Keeping this tight avoids seeing the robot body.
ROI_TOP      = 0.72    # start of ROI (72 % down the frame)
ROI_BOT      = 0.97    # end of ROI  (97 % — leave a sliver at bottom)

# ── Multi-band scan within ROI ───────────────────────────────────
# ROI is split into NEAR (bottom), MID (middle) and FAR (top) thirds.
# NEAR drives immediate correction; FAR drives look-ahead.
NEAR_WEIGHT  =  3      # weight for near-band in combined offset
MID_WEIGHT   =  2
FAR_WEIGHT   =  1
ANGLE_GAIN   =  8.0    # how much line angle influences steering (degrees)

# ── Quality filters ──────────────────────────────────────────────
MIN_WHITE_PX =  300    # minimum white pixels per band to trust it
MIN_ASPECT   =  1.5    # minimum contour width/height ratio (line-like shape)

# ── PID gains ────────────────────────────────────────────────────
KP           = 35.0
KI           =  0.5
KD           =  4.0

# ── Speed ────────────────────────────────────────────────────────
FOLLOW_SPEED =  13     # normal speed when line is centred and clear
CORRECT_SPEED=  10     # speed during moderate correction
SEEK_SPEED   =   7     # creep speed in gap / low confidence

# Confidence → speed mapping
# confidence = white_px / CONF_SCALE  (clamped 0..1)
CONF_SCALE   = 3000    # white_px for "full confidence"

# ── Steering ─────────────────────────────────────────────────────
MAX_STEER    = 30.0    # servo hard limit
STEER_RATE   = 15.0    # max degrees change per frame — faster response

# ── Gap tolerance ────────────────────────────────────────────────
GAP_FRAMES   =  6      # frames with no detection before entering seek mode
GAP_DECAY    =  0.99   # decay per gap frame — high = keeps turning toward last line
GAP_BIAS     =  0.70   # fraction of last offset applied as steering bias in gap

# ── Obstacle ─────────────────────────────────────────────────────
STOP_DIST    = 15      # cm
SLOW_DIST    = 35      # cm

# ── Loop ─────────────────────────────────────────────────────────
LOOP_HZ      = 30
PRINT_EVERY  = 10

# ════════════════════════════════════════════════════════════════
#  INIT
# ════════════════════════════════════════════════════════════════

px = Picarx()


# ════════════════════════════════════════════════════════════════
#  FOLLOWER
# ════════════════════════════════════════════════════════════════

class CameraFollower:

    def __init__(self):
        self._steer     =  0.0
        self._i_err     =  0.0
        self._prev_err  =  0.0
        self._gap_cnt   =  0
        self._last_off  =  0.0
        self._last_ang  =  0.0
        self._obs_cnt   =  0
        self._obs_stuck =  0

    # ── Smooth steering ──────────────────────────────────────────
    def _apply_steer(self, target):
        delta = max(-STEER_RATE, min(STEER_RATE, target - self._steer))
        self._steer = max(-MAX_STEER, min(MAX_STEER, self._steer + delta))
        return self._steer

    def _speed_cap(self, dist, base):
        if 0 < dist < SLOW_DIST:
            return min(base, SEEK_SPEED)
        return base

    # ── White mask with adaptive threshold ───────────────────────
    def _white_mask(self, bgr_roi):
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        mean_v = float(hsv[:, :, 2].mean())
        v_min  = int(WHITE_V_MIN + ADAPTIVE_K * (mean_v - 128))
        v_min  = max(100, min(240, v_min))
        mask = cv2.inRange(
            hsv,
            np.array([0,           0, v_min], np.uint8),
            np.array([180, WHITE_S_MAX,  255], np.uint8),
        )
        # Remove single-pixel noise
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return mask, v_min

    # ── Band centre-of-mass ──────────────────────────────────────
    @staticmethod
    def _band_cx(mask_band, w):
        """Weighted column centre for a horizontal band. Returns (cx, px_count)."""
        col_sum = mask_band.sum(axis=0).astype(float)
        total   = col_sum.sum()
        if total < MIN_WHITE_PX * 255:
            return None, 0
        cx = float(np.dot(np.arange(w), col_sum)) / total
        return cx, int(total // 255)

    # ── Main detection ───────────────────────────────────────────
    def detect_line(self, frame):
        """
        Returns (offset, angle, total_px, v_thresh, debug_frame).

        offset : -1.0 … +1.0  horizontal position of line (+ = right)
        angle  : signed slope of line across ROI bands (+ = heading right)
        total_px: total white pixels found (proxy for confidence)
        v_thresh: adaptive V threshold actually used this frame
        """
        h, w    = frame.shape[:2]
        roi_y0  = int(h * ROI_TOP)
        roi_y1  = int(h * ROI_BOT)
        roi     = frame[roi_y0:roi_y1, :]
        roi_h   = roi_y1 - roi_y0

        mask, v_thresh = self._white_mask(roi)

        # Split ROI into 3 bands (bottom=near, top=far)
        b1 = roi_h * 2 // 3   # near:  rows b1..roi_h
        b2 = roi_h * 1 // 3   # far:   rows 0..b2

        near_cx, near_px = self._band_cx(mask[b1:,   :], w)
        mid_cx,  mid_px  = self._band_cx(mask[b2:b1, :], w)
        far_cx,  far_px  = self._band_cx(mask[:b2,   :], w)

        total_px = near_px + mid_px + far_px

        # Build weighted offset from available bands
        parts, weights = [], []
        if near_cx is not None:
            parts.append(near_cx); weights.append(NEAR_WEIGHT)
        if mid_cx  is not None:
            parts.append(mid_cx);  weights.append(MID_WEIGHT)
        if far_cx  is not None:
            parts.append(far_cx);  weights.append(FAR_WEIGHT)

        if not parts:
            return 0.0, 0.0, 0, v_thresh, frame

        total_w  = sum(weights)
        cx_w     = sum(p * wt for p, wt in zip(parts, weights)) / total_w
        offset   = (cx_w - w / 2.0) / (w / 2.0)

        # Line angle: how much does it drift left/right from near→far?
        angle = 0.0
        if near_cx is not None and far_cx is not None:
            # positive = line is heading right as we look ahead
            angle = (far_cx - near_cx) / (w / 2.0)

        # ── Annotate frame ───────────────────────────────────────
        cx_px = int((offset + 1.0) / 2.0 * w)
        cv2.rectangle(frame, (0, roi_y0), (w, roi_y1), (0, 200, 200), 1)
        cv2.line(frame, (w // 2, roi_y0), (w // 2, roi_y1), (255, 0, 0), 1)
        if near_cx is not None:
            cv2.circle(frame, (int(near_cx), roi_y0 + roi_h * 5 // 6), 5, (0, 255, 0),   -1)
        if mid_cx  is not None:
            cv2.circle(frame, (int(mid_cx),  roi_y0 + roi_h // 2),     5, (0, 200, 255), -1)
        if far_cx  is not None:
            cv2.circle(frame, (int(far_cx),  roi_y0 + roi_h // 6),     5, (0, 100, 255), -1)

        return offset, angle, total_px, v_thresh, frame

    # ── Decision ─────────────────────────────────────────────────
    def decide(self, frame, dist):
        """Returns (steer_deg, speed_pct, label, annotated_frame)."""

        # ── Obstacle ─────────────────────────────────────────────
        if 0 < dist < STOP_DIST:
            self._obs_cnt += 1
        else:
            self._obs_cnt = 0

        if self._obs_cnt >= 3:
            self._obs_stuck += 1
            if self._obs_stuck > 50:
                self._obs_stuck = 0
                steer = self._apply_steer(-MAX_STEER * (1 if self._last_off >= 0 else -1))
                return steer, -SEEK_SPEED, 'OBSTACLE/reverse_escape', frame
            return 0.0, 0, f'OBSTACLE/stop ({dist:.0f}cm)', frame
        else:
            self._obs_stuck = 0

        offset, angle, total_px, v_thr, frame = self.detect_line(frame)

        # ── Gap between dashes ────────────────────────────────────
        if total_px < MIN_WHITE_PX:
            self._gap_cnt  += 1
            self._i_err     = 0.0
            self._last_off *= GAP_DECAY
            self._last_ang *= GAP_DECAY
            if self._gap_cnt >= GAP_FRAMES:
                # Extrapolate: continue with decayed last known position
                target = self._last_off * MAX_STEER * GAP_BIAS
                steer  = self._apply_steer(target)
                speed  = self._speed_cap(dist, SEEK_SPEED)
                return steer, speed, (f'GAP/seek ({self._gap_cnt})'
                                      f' bias={target:+.1f}°'), frame
            return self._steer, self._speed_cap(dist, CORRECT_SPEED), 'GAP/glitch', frame

        # Line found
        self._gap_cnt  = 0
        self._last_off = offset
        self._last_ang = angle

        # ── Angle-aware PID ───────────────────────────────────────
        # Error = where line is NOW + where it's HEADING
        err            = offset + ANGLE_GAIN * angle / MAX_STEER
        self._i_err    = max(-1.5, min(1.5, self._i_err + err))
        d_err          = err - self._prev_err
        self._prev_err = err

        raw_steer = KP * err + KI * self._i_err + KD * d_err
        steer     = self._apply_steer(raw_steer)

        # ── Confidence → speed ────────────────────────────────────
        conf  = min(1.0, total_px / CONF_SCALE)
        if conf > 0.7 and abs(offset) < 0.25:
            speed = self._speed_cap(dist, FOLLOW_SPEED)   # centred and confident
        elif abs(offset) > 0.6:
            speed = self._speed_cap(dist, SEEK_SPEED)     # big correction — crawl
        elif conf > 0.4:
            speed = self._speed_cap(dist, CORRECT_SPEED)
        else:
            speed = self._speed_cap(dist, SEEK_SPEED)

        label = (f'LINE off={offset:+.2f} ang={angle:+.2f}'
                 f' conf={conf:.2f} steer={steer:+.1f}° V>{v_thr}')
        return steer, speed, label, frame


# ════════════════════════════════════════════════════════════════
#  CALIBRATION
# ════════════════════════════════════════════════════════════════

def run_calibration(cam, seconds=12):
    print(f"\n{'═'*64}")
    print(" CAMERA CALIBRATION — move robot over each surface")
    print(f"{'═'*64}")
    print(f" WHITE_V_MIN={WHITE_V_MIN}  WHITE_S_MAX={WHITE_S_MAX}")
    print(f" ROI = rows {ROI_TOP*100:.0f}%–{ROI_BOT*100:.0f}% of frame")
    print()
    print(" White dash  → white_px should be HIGH  (>300 per band)")
    print(" Dark foam   → white_px should be ZERO")
    print(" Grey carpet → white_px should be low or zero")
    print()
    sleep(1.0)
    t_end = monotonic() + seconds
    while monotonic() < t_end:
        frame     = cam.capture_array()
        h, w      = frame.shape[:2]
        roi       = frame[int(h*ROI_TOP):int(h*ROI_BOT), :]
        hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_v    = float(hsv[:, :, 2].mean())
        v_thr     = int(WHITE_V_MIN + ADAPTIVE_K * (mean_v - 128))
        v_thr     = max(100, min(240, v_thr))
        mask      = cv2.inRange(hsv,
                                np.array([0,           0, v_thr], np.uint8),
                                np.array([180, WHITE_S_MAX,  255], np.uint8))
        wpx       = int(mask.sum() // 255)
        cy, cx    = roi.shape[0]//2, roi.shape[1]//2
        hc        = hsv[cy, cx]
        rem       = t_end - monotonic()
        print(f"  [{rem:4.1f}s]  ROI-centre HSV=({hc[0]:3d},{hc[1]:3d},{hc[2]:3d})"
              f"  mean_V={mean_v:.0f}  V_thr={v_thr}"
              f"  white_px={wpx:5d}",
              end='\r', flush=True)
        sleep(0.12)
    print(f"\n\n Done.\n{'═'*64}\n")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    global WHITE_V_MIN, WHITE_S_MAX, CAM_TILT

    import argparse
    p = argparse.ArgumentParser(description='Picar-X camera line follower v2')
    p.add_argument('--calibrate',   action='store_true')
    p.add_argument('--cam-tilt',    type=float, default=CAM_TILT,
                   help=f'Camera tilt degrees, negative=down (default {CAM_TILT})')
    p.add_argument('--white-v',     type=int,   default=WHITE_V_MIN,
                   help=f'Base HSV V threshold (default {WHITE_V_MIN})')
    p.add_argument('--white-s',     type=int,   default=WHITE_S_MAX,
                   help=f'Max HSV S threshold (default {WHITE_S_MAX})')
    p.add_argument('--save-frames', action='store_true',
                   help='Save annotated frames to /tmp/cam_debug/')
    args = p.parse_args()

    WHITE_V_MIN = args.white_v
    WHITE_S_MAX = args.white_s
    CAM_TILT    = args.cam_tilt

    print(f"\n{'═'*60}")
    print(" Picar-X Camera Line Follower  v2")
    print(f"{'═'*60}")
    print(f" CAM_TILT={CAM_TILT}°  WHITE_V_MIN={WHITE_V_MIN}  WHITE_S_MAX={WHITE_S_MAX}")
    print(f" ROI={ROI_TOP*100:.0f}%-{ROI_BOT*100:.0f}%  RES={CAM_W}x{CAM_H}")
    print(f" FOLLOW={FOLLOW_SPEED}%  SEEK={SEEK_SPEED}%  STOP_DIST={STOP_DIST}cm")
    print(f"{'═'*60}\n")

    print(" Initialising camera...")
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={'size': (CAM_W, CAM_H), 'format': 'BGR888'},
        controls={'FrameRate': CAM_FPS},
    )
    cam.configure(cfg)
    cam.start()
    sleep(1.5)
    print(f" Camera ready: {CAM_W}x{CAM_H} @ {CAM_FPS}fps")

    px.set_cam_pan_angle(CAM_PAN)
    px.set_cam_tilt_angle(CAM_TILT)
    sleep(0.5)
    print(f" Camera aimed: tilt={CAM_TILT}°\n")

    if args.save_frames:
        os.makedirs('/tmp/cam_debug', exist_ok=True)
        print(" Debug frames → /tmp/cam_debug/\n")

    if args.calibrate:
        run_calibration(cam, seconds=12)
        cam.stop(); px.stop()
        return

    follower   = CameraFollower()
    px.set_dir_servo_angle(0)

    print(" Starting.  Ctrl+C to stop.\n")
    print(f" {'State':56s} {'Steer':>6} {'Spd':>4} {'Dist':>5}")
    print(f" {'-'*56} {'-'*6} {'-'*4} {'-'*5}")

    cycle      = 0
    save_cycle = 0
    stopped    = False
    period     = 1.0 / LOOP_HZ

    try:
        while True:
            t0    = monotonic()
            frame = cam.capture_array()
            dist  = px.get_distance()

            steer, speed, label, ann = follower.decide(frame, dist)

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
                print(f" {label:56s} {steer:+5.0f}° {speed:3d}%  {dist_s}", flush=True)

            if args.save_frames and cycle % 15 == 0:
                cv2.imwrite(f'/tmp/cam_debug/f{save_cycle:05d}.jpg', ann)
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
        print(" Done.\n")


if __name__ == '__main__':
    main()

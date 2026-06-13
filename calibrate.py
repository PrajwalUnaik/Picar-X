#!/usr/bin/env python3
"""
Picar-X Calibration Tool
========================
Calibrates steering servo center, camera pan center, and camera tilt angle
using a live USB webcam feed (view at http://<pi-ip>:8080) and terminal keys.

Controls during each step:
  +  /  =    increase offset by 1°
  -          decrease offset by 1°
  ]          increase offset by 5°
  [          decrease offset by 5°
  0          reset offset to 0°
  s / Enter  save and advance to next step
  q          quit without saving remaining steps

Run:
  python3 calibrate.py
  python3 calibrate.py --cam-index 8
"""

import argparse
import sys
import threading
from time import sleep

import cv2
import flask
from picamera2 import Picamera2

try:
    from picarx import Picarx
except ImportError:
    print("ERROR: picarx not found — is robot-hat installed?")
    sys.exit(1)

# ── Web stream (same pattern as lane_follower_v2) ─────────────────────────────
_stream_frame = None
_stream_lock  = threading.Lock()
_app          = flask.Flask(__name__)

@_app.route('/')
def _index():
    return '<html><body><h2>Picar-X Calibration</h2><img src="/stream"><br><pre id="s"></pre></body></html>'

@_app.route('/stream')
def _stream():
    def _gen():
        while True:
            with _stream_lock:
                f = _stream_frame
            if f is not None:
                ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
            sleep(0.05)
    return flask.Response(_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def _push(frame):
    global _stream_frame
    with _stream_lock:
        _stream_frame = frame

def _start_stream():
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    _app.run(host='0.0.0.0', port=8080, threaded=True)

# ── Pi Camera ─────────────────────────────────────────────────────────────────
def open_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={'size': (640, 480), 'format': 'BGR888'},
        controls={'FrameRate': 30},
    )
    cam.configure(cfg)
    cam.start()
    sleep(1.5)
    print(' Pi camera opened: 640×480 @ 30fps')
    return cam


# ── Calibration steps ─────────────────────────────────────────────────────────

def _overlay(frame, title: str, offset: float, hint: str = ''):
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    cv2.putText(out, f'offset = {offset:+.1f} deg', (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    if hint:
        cv2.putText(out, hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return out


def calibrate_servo(cam, px, name: str, set_fn, save_fn,
                    tip: str, init_offset: float = 0.0) -> float:
    """
    Interactive calibration using plain input().
    set_fn(offset)  — moves servo to raw offset angle
    save_fn(offset) — persists offset to picarx config
    """
    offset = init_offset
    set_fn(offset)

    print(f'\n{"─"*60}')
    print(f' Calibrating: {name}')
    print(f' {tip}')
    print(f' Commands:')
    print(f'   <number>   set offset in degrees (e.g. 3, -2.5)')
    print(f'   +<number>  add to current (e.g. +1, +5)')
    print(f'   -<number>  subtract from current (e.g. -1, -5)')
    print(f'   0          reset to 0')
    print(f'   s          save and go to next step')
    print(f'   q          skip without saving')
    print(f'{"─"*60}')
    print(f' Current offset: {offset:+.1f}°')
    print(f' (Live feed at http://<pi-ip>:8080)\n')

    # Push camera frames in background while waiting for input
    def _push_loop():
        while _push_loop.running:
            frame = cam.capture_array()
            if frame is not None:
                disp = _overlay(frame, f'CALIB: {name}', offset,
                                'type number/+/-/0/s/q then Enter')
                _push(disp)
            sleep(0.05)
    _push_loop.running = True
    t = threading.Thread(target=_push_loop, daemon=True)
    t.start()

    try:
        while True:
            try:
                cmd = input(f'  [{name}] offset={offset:+.1f}° > ').strip()
            except EOFError:
                break

            if cmd in ('s', 'S', ''):
                save_fn(offset)
                print(f' ✓ {name} saved: {offset:+.1f}°')
                return offset
            elif cmd in ('q', 'Q'):
                print(f' Skipped {name} (not saved).')
                return offset
            elif cmd == '0':
                offset = 0.0
            else:
                try:
                    if cmd.startswith('+') and len(cmd) > 1:
                        offset += float(cmd[1:])
                    elif cmd.startswith('-') and len(cmd) > 1:
                        offset -= float(cmd[1:])
                    else:
                        offset = float(cmd)
                except ValueError:
                    print(f'  Unknown: "{cmd}" — type a number, +N, -N, s, or q')
                    continue

            set_fn(offset)
            print(f'  -> {name} = {offset:+.1f}')
    finally:
        _push_loop.running = False
        t.join(timeout=0.5)
    return offset


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Picar-X Calibration Tool')
    ap.add_argument('--skip-steering', action='store_true',
                    help='Skip steering servo calibration')
    ap.add_argument('--skip-pan',      action='store_true',
                    help='Skip camera pan calibration')
    ap.add_argument('--skip-tilt',     action='store_true',
                    help='Skip camera tilt calibration')
    args = ap.parse_args()

    print(f"\n{'═'*60}")
    print(' Picar-X Calibration Tool')
    print(f'{"═"*60}')
    print(' Open http://<pi-ip>:8080 in a browser to see the camera feed.')
    print(' Use terminal keys to adjust each servo.\n')

    px  = Picarx()
    cam = open_camera()

    # Start web stream
    t = threading.Thread(target=_start_stream, daemon=True)
    t.start()
    sleep(0.5)

    results = {}

    try:
        # ── Step 1: Steering servo ────────────────────────────────────────────
        if not args.skip_steering:
            # Reset pan/tilt so camera faces forward — easier to judge straight
            px.set_cam_pan_angle(0)
            px.set_cam_tilt_angle(0)
            # Start from current saved value
            init = px.dir_cali_val
            results['steering'] = calibrate_servo(
                cam, px,
                name       = 'Steering center',
                set_fn     = px.dir_servo_calibrate,
                save_fn    = px.dir_servo_calibrate,
                tip        = 'Adjust until the front wheels point perfectly straight.',
                init_offset= init,
            )
        else:
            print(' [SKIP] Steering calibration skipped.')

        # ── Step 2: Camera pan ────────────────────────────────────────────────
        if not args.skip_pan:
            px.set_dir_servo_angle(0)
            init = px.cam_pan_cali_val
            results['cam_pan'] = calibrate_servo(
                cam, px,
                name       = 'Camera pan center',
                set_fn     = px.cam_pan_servo_calibrate,
                save_fn    = px.cam_pan_servo_calibrate,
                tip        = 'Adjust until the camera points exactly forward (no left/right lean).',
                init_offset= init,
            )
        else:
            print(' [SKIP] Camera pan calibration skipped.')

        # ── Step 3: Camera tilt ───────────────────────────────────────────────
        if not args.skip_tilt:
            px.set_dir_servo_angle(0)
            # Show a sensible starting tilt (-15° is the lane follower default)
            init = px.cam_tilt_cali_val if px.cam_tilt_cali_val != 0 else -15.0
            px.cam_tilt_servo_calibrate(init)
            results['cam_tilt'] = calibrate_servo(
                cam, px,
                name       = 'Camera tilt',
                set_fn     = px.cam_tilt_servo_calibrate,
                save_fn    = px.cam_tilt_servo_calibrate,
                tip        = ('Adjust tilt so the lane lines are visible in the lower 2/3 '
                              'of the frame. -15° is a good starting point.'),
                init_offset= init,
            )
        else:
            print(' [SKIP] Camera tilt calibration skipped.')

    finally:
        # Ensure car is safe
        px.stop()
        px.set_dir_servo_angle(0)
        cam.stop()

    print(f'\n{"═"*60}')
    print(' Calibration complete. Saved values:')
    for k, v in results.items():
        print(f'   {k}: {v:+.1f}°')
    print()
    print(' These offsets are stored in /opt/picar-x/picar-x.conf and')
    print(' will be loaded automatically by lane_follower_v2.py.')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Picar-X White Dashed-Line Follower
------------------------------------
Track environment:
  - Centre line : dashed white tape (what we follow)
  - Road surface: dark foam mats
  - Off-road    : light grey carpet

Sensor array (front-mounted, pointing down):
  A0 = Left sensor
  A1 = Middle sensor
  A2 = Right sensor

Strategy:
  - Keep the MIDDLE sensor over the white dashed line
  - Steer left/right based on which sensors see white
  - When NO sensor sees white (gap between dashes): slow to a creep
  - Obstacle: full stop
"""

from picarx import Picarx
from time import sleep, monotonic
import sys

# ════════════════════════════════════════════════════════════════
#  TUNING
# ════════════════════════════════════════════════════════════════

WHITE_THRESH  = 1300   # ADC >= this → white line detected

# ── Speeds ───────────────────────────────────────────────────────
FOLLOW_SPEED  = 11     # normal line-following speed
CORRECT_SPEED =  9     # speed while steering correction active
SEEK_SPEED    =  7     # slow creep while searching for next dash

# ── Steering ─────────────────────────────────────────────────────
STEER_GENTLE  = 12     # line slightly off-centre (M + one side white)
STEER_SHARP   = 25     # line well off-centre (one side only white)
STEER_RATE    =  6     # max degrees change per 20 ms cycle

# ── Straight-line bias ───────────────────────────────────────────
# Negative corrects a right-drift; tune via --straight-offset flag.
STRAIGHT_OFFSET = 0.0

# ── Gap detection ────────────────────────────────────────────────
# How many consecutive no-white cycles (at 50 Hz) before we slow to
# a creep. 4 cycles ≈ 80 ms.
GAP_CYCLES    =  4

# ── Obstacle ─────────────────────────────────────────────────────
STOP_DIST     = 15     # cm — full stop
SLOW_DIST     = 35     # cm — reduce to SEEK_SPEED

# ── Loop ─────────────────────────────────────────────────────────
LOOP_HZ       = 50
LOOP_PERIOD   = 1.0 / LOOP_HZ
PRINT_EVERY   = 25


# ════════════════════════════════════════════════════════════════
#  INIT
# ════════════════════════════════════════════════════════════════

px = Picarx()
sleep(0.3)


# ════════════════════════════════════════════════════════════════
#  LINE FOLLOWER — stateful decision engine
# ════════════════════════════════════════════════════════════════

class LineFollower:
    """
    Steers to keep the middle sensor over the white dashed centre line.
    Slows to a creep when in the gap between dashes.
    """

    def __init__(self):
        self._steer      = 0.0
        self.last_dir    = 0      # last correction: +1 right, -1 left, 0 none
        self._gap_count  = 0      # consecutive no-white cycles
        self._obs_consec = 0      # consecutive obstacle readings
        self._obs_stuck  = 0      # cycles spent stopped at obstacle

    def _apply_steer(self, target):
        delta = target - self._steer
        delta = max(-STEER_RATE, min(STEER_RATE, delta))
        self._steer += delta
        self._steer  = max(-30.0, min(30.0, self._steer))
        return self._steer

    def _speed_cap(self, dist, base):
        if 0 < dist < SLOW_DIST:
            return min(base, SEEK_SPEED)
        return base

    def decide(self, adc, dist):
        """
        Args:
            adc  : [L, M, R] raw ADC values
            dist : ultrasonic distance in cm

        Returns:
            (steer_deg, speed_pct, label_str)
        """
        L, M, R = adc

        Lw = (L >= WHITE_THRESH)
        Mw = (M >= WHITE_THRESH)
        Rw = (R >= WHITE_THRESH)
        any_white = Lw or Mw or Rw

        # ── 1. Obstacle ───────────────────────────────────────────
        if 0 < dist < STOP_DIST:
            self._obs_consec += 1
        else:
            self._obs_consec = 0

        if self._obs_consec >= 3:
            self._obs_stuck += 1
            if self._obs_stuck > 50:   # 1 s stuck → reverse to escape
                steer = self._apply_steer(-STEER_SHARP * (self.last_dir or 1))
                self._obs_stuck = 0
                return steer, -SEEK_SPEED, 'OBSTACLE/reverse_escape'
            return 0.0, 0, f'OBSTACLE/stop ({self._obs_consec})'
        else:
            self._obs_stuck = 0

        # ── 2. Gap between dashes ─────────────────────────────────
        if not any_white:
            self._gap_count += 1
            if self._gap_count >= GAP_CYCLES:
                # Hold last steer; creep forward slowly to find next dash
                steer = self._apply_steer(STRAIGHT_OFFSET + self.last_dir * STEER_GENTLE * 0.3)
                speed = self._speed_cap(dist, SEEK_SPEED)
                return steer, speed, f'GAP/seeking ({self._gap_count})'
            else:
                # Brief glitch — don't react yet
                steer = self._apply_steer(self._steer)
                return steer, self._speed_cap(dist, FOLLOW_SPEED), 'GAP/glitch'

        # White detected — reset gap counter
        self._gap_count = 0

        cap = self._speed_cap(dist, FOLLOW_SPEED)

        # ── 3. Line position → steering ───────────────────────────

        # Perfect: only middle sees white → dead-centre
        if Mw and not Lw and not Rw:
            steer = self._apply_steer(STRAIGHT_OFFSET)
            return steer, cap, 'LINE/centre'

        # Middle + left white: line drifting left → correct right
        if Mw and Lw and not Rw:
            self.last_dir = 1
            steer = self._apply_steer(STRAIGHT_OFFSET + STEER_GENTLE)
            return steer, self._speed_cap(dist, CORRECT_SPEED), 'LINE/left_of_centre→right'

        # Middle + right white: line drifting right → correct left
        if Mw and not Lw and Rw:
            self.last_dir = -1
            steer = self._apply_steer(STRAIGHT_OFFSET - STEER_GENTLE)
            return steer, self._speed_cap(dist, CORRECT_SPEED), 'LINE/right_of_centre→left'

        # All three white: wide stripe or junction → continue last direction
        if Lw and Mw and Rw:
            steer = self._apply_steer(STRAIGHT_OFFSET)
            return steer, cap, 'LINE/wide_stripe'

        # Only left white: line is to the left → steer left sharply
        if Lw and not Mw and not Rw:
            self.last_dir = -1
            steer = self._apply_steer(STRAIGHT_OFFSET - STEER_SHARP)
            return steer, self._speed_cap(dist, CORRECT_SPEED), 'LINE/lost_right→sharp_left'

        # Only right white: line is to the right → steer right sharply
        if not Lw and not Mw and Rw:
            self.last_dir = 1
            steer = self._apply_steer(STRAIGHT_OFFSET + STEER_SHARP)
            return steer, self._speed_cap(dist, CORRECT_SPEED), 'LINE/lost_left→sharp_right'

        # Left + right but not middle (straddling a very narrow dash)
        if Lw and not Mw and Rw:
            steer = self._apply_steer(STRAIGHT_OFFSET)
            return steer, cap, 'LINE/straddle'

        # Fallback
        steer = self._apply_steer(STRAIGHT_OFFSET)
        return steer, cap, f'LINE/fallback L={L} M={M} R={R}'


# ════════════════════════════════════════════════════════════════
#  CALIBRATION MODE
# ════════════════════════════════════════════════════════════════

def run_calibration(seconds=8):
    print(f"\n{'═'*60}")
    print(" SENSOR CALIBRATION CHECK")
    print(f"{'═'*60}")
    print(f" WHITE_THRESH = {WHITE_THRESH}")
    print(" Aim to see  W  when sensors are over a white dash,")
    print("             -  (below threshold) everywhere else.\n")
    sleep(1.0)
    t_end = monotonic() + seconds
    while monotonic() < t_end:
        adc = px.get_grayscale_data()
        cls = ['W' if v >= WHITE_THRESH else '-' for v in adc]
        bar = ''.join(cls)
        remaining = t_end - monotonic()
        print(f"  [{remaining:4.1f}s]  ADC: {adc[0]:4d} {adc[1]:4d} {adc[2]:4d}  "
              f"class: {bar}  (L M R)",
              end='\r', flush=True)
        sleep(0.15)
    print(f"\n\n Calibration check complete.\n{'═'*60}\n")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    global WHITE_THRESH, STRAIGHT_OFFSET
    import argparse
    parser = argparse.ArgumentParser(description='Picar-X white dashed-line follower')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run sensor calibration check then exit')
    parser.add_argument('--white-thresh', type=int, default=WHITE_THRESH,
                        help=f'ADC threshold for white line (default {WHITE_THRESH})')
    parser.add_argument('--straight-offset', type=float, default=STRAIGHT_OFFSET,
                        help='Steering bias when centred, degrees. Negative corrects right-drift.')
    args = parser.parse_args()

    WHITE_THRESH    = args.white_thresh
    STRAIGHT_OFFSET = args.straight_offset

    print(f"\n{'═'*60}")
    print(" Picar-X White Line Follower")
    print(f"{'═'*60}")
    print(f" WHITE_THRESH={WHITE_THRESH}  STRAIGHT_OFFSET={STRAIGHT_OFFSET:+.1f}°")
    print(f" FOLLOW={FOLLOW_SPEED}%  SEEK={SEEK_SPEED}%  STOP_DIST={STOP_DIST}cm")
    print(f"{'═'*60}\n")

    if args.calibrate:
        run_calibration(seconds=10)
        px.stop()
        return

    follower = LineFollower()
    px.set_dir_servo_angle(0)
    sleep(0.3)

    print(" Starting.  Ctrl+C to stop.\n")
    print(f" {'State':42s} {'ADC-L':>5} {'ADC-M':>5} {'ADC-R':>5} {'Steer':>6} {'Spd':>4} {'Dist':>6}")
    print(f" {'-'*42} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*4} {'-'*6}")

    cycle   = 0
    stopped = False

    try:
        while True:
            t0 = monotonic()

            adc  = px.get_grayscale_data()
            dist = px.get_distance()

            steer, speed, label = follower.decide(adc, dist)

            px.set_dir_servo_angle(steer)

            if speed == 0:
                if not stopped:
                    px.stop()
                    stopped = True
            elif speed < 0:
                px.backward(abs(speed))
                stopped = False
            else:
                px.forward(speed)
                stopped = False

            if cycle % PRINT_EVERY == 0:
                dist_str = f'{dist:.0f}cm' if dist > 0 else '  --- '
                print(f" {label:42s} {adc[0]:5d} {adc[1]:5d} {adc[2]:5d} "
                      f"{steer:+5.0f}° {speed:3d}%  {dist_str:>6}", flush=True)

            cycle += 1

            elapsed = monotonic() - t0
            wait    = LOOP_PERIOD - elapsed
            if wait > 0:
                sleep(wait)

    except KeyboardInterrupt:
        print("\n\n Ctrl+C — stopping.")
    except Exception as e:
        print(f"\n\n ERROR: {e}")
        raise
    finally:
        px.stop()
        px.set_dir_servo_angle(0)
        print(" Motors stopped.\n")


if __name__ == '__main__':
    main()

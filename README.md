# Picar-X Autonomous Line Follower

Autonomous white dashed-line follower for the **SunFounder Picar-X** robot car, running on a Raspberry Pi. Two implementations are provided — a fast grayscale-sensor version and a more accurate camera + OpenCV version.

---

## Hardware

| Component | Details |
|---|---|
| Robot | SunFounder Picar-X |
| Platform | Raspberry Pi (aarch64, hostname `pi-six`) |
| MCU | STM32-type onboard HAT at I2C 0x14 (PWM / ADC) |
| Sensors | 3× grayscale (A0=Left, A1=Middle, A2=Right) |
| Camera | OV5647, controlled via picamera2 |
| Distance | HC-SR04 ultrasonic (TRIG=D2, ECHO=D3) |

---

## Track Environment

```
  ┌─────────────────────────────────────────┐
  │  Grey carpet (off-road)                 │
  │  ┌───────────────────────────────────┐  │
  │  │  Dark foam road surface           │  │
  │  │    - - - - - - - - - - -          │  │  ← white dashed centre line
  │  │                                   │  │
  │  └───────────────────────────────────┘  │
  │  Grey carpet (off-road)                 │
  └─────────────────────────────────────────┘
```

- **White dashed line** — the centre line to follow
- **Dark foam** — the road surface
- **Grey carpet** — off-road area to avoid

---

## Files

| File | Description |
|---|---|
| `road_follower.py` | Grayscale sensor-based line follower |
| `cam_follower.py` | Camera + OpenCV + PID line follower (recommended) |

---

## `road_follower.py` — Grayscale Sensor Follower

Uses the 3 downward-facing grayscale sensors to detect the white dashed line.

### How it works

Each sensor returns an ADC value (0–4095):
- **White tape**: ~1300–3600
- **Dark foam**: ~200–400
- **Grey carpet**: ~700–1500

The `LineFollower` class maps sensor patterns to steering decisions:

| Left | Mid | Right | Action |
|---|---|---|---|
| – | W | – | Straight (perfectly centred) |
| W | W | – | Gentle right (line drifting left) |
| – | W | W | Gentle left (line drifting right) |
| W | – | – | Sharp left (line almost lost) |
| – | – | W | Sharp right (line almost lost) |
| – | – | – | GAP: slow creep until next dash |

### Usage

```bash
# Run
python3 road_follower.py

# Calibrate sensor thresholds
python3 road_follower.py --calibrate

# Adjust white detection threshold
python3 road_follower.py --white-thresh 1400

# Correct straight-line drift (negative = correct right-drift)
python3 road_follower.py --straight-offset -2.0
```

### Key tuning constants

```python
WHITE_THRESH  = 1300   # ADC >= this → white line
FOLLOW_SPEED  = 11     # % — normal following speed
SEEK_SPEED    = 7      # % — creep speed in gaps between dashes
STEER_GENTLE  = 12     # degrees — gentle correction
STEER_SHARP   = 25     # degrees — sharp correction
GAP_CYCLES    = 4      # frames with no white before seeking
```

---

## `cam_follower.py` — Camera + OpenCV Follower (Recommended)

Uses the front camera tilted downward to detect the white line with OpenCV, then steers with a PID controller. Far more robust than the grayscale sensors.

### How it works

1. **Camera** captures 320×240 BGR frames at 30fps
2. **ROI** — only the bottom 45% of the frame (road closest to robot) is analysed
3. **HSV threshold** — white pixels: Saturation < 50, Value > 190
4. **Column centre of mass** — finds the horizontal centre of all white pixels → `offset` (-1.0 to +1.0)
5. **PID controller** — converts offset to steering angle
6. **Gap detection** — when no white pixels found for 8+ frames, slow to seek speed

```
Frame (320×240):
┌────────────────────────────────────────┐
│                                        │  ← ignored (sky/background)
│                                        │
├────────────────────────────────────────┤  ← ROI_TOP (55%)
│         [analysed region]              │
│              ●  ← detected line centre │
│     blue │   green dot                 │
└────────────────────────────────────────┘
```

PID formula:
```
steer = KP × error + KI × Σerror + KD × Δerror
```
Where `error = offset` (how far the line is from frame centre).

### Usage

```bash
# Run with camera
python3 cam_follower.py

# Calibrate colour thresholds (point at white then grey)
python3 cam_follower.py --calibrate

# Adjust camera tilt angle (more negative = looks further down)
python3 cam_follower.py --cam-tilt -25

# Save annotated frames for debugging
python3 cam_follower.py --save-frames
# Frames saved to /tmp/cam_debug/fNNNNN.jpg

# Override colour thresholds
python3 cam_follower.py --white-v 180 --white-s 60
```

### Key tuning constants

```python
CAM_TILT     = -30.0   # degrees — camera tilt (negative = look down)
WHITE_V_MIN  = 190     # minimum HSV brightness for white
WHITE_S_MAX  =  50     # maximum HSV saturation for white
ROI_TOP      = 0.55    # use bottom 45% of frame
MIN_WHITE_PX = 200     # minimum pixels to count as "line found"
KP           = 28.0    # PID proportional gain
KI           =  0.8    # PID integral gain
KD           =  4.0    # PID derivative gain
FOLLOW_SPEED =  11     # % — normal speed
SEEK_SPEED   =   7     # % — gap creep speed
```

### Tuning guide

**Camera angle** (`--cam-tilt`): Start at -30°. If the robot can't see the line at all, try -25°. If it sees too far ahead and misses nearby line, try -35°.

**White threshold** (`--white-v`, `--white-s`): Run `--calibrate` and note the HSV V value of the white dash. Set `WHITE_V_MIN` about 20 below that value. Ensure grey carpet gives a V value well below `WHITE_V_MIN`.

**PID gains**: If the robot oscillates (wobbles left-right), reduce `KP` or increase `KD`. If it's slow to correct, increase `KP`.

---

## Calibration

### Direction servo (straight-line drift)

If the robot drifts left or right when going straight:

```bash
python3 /home/admin/picar-x/example/1.cali_servo_motor.py
```

- Press `1` to select the direction servo
- Press `W`/`S` to increase/decrease the offset
- Press `SPACE` then `Y` to save

Current calibration stored in `/opt/picar-x/picar-x.conf`.

### Motor direction

If a motor runs backwards (robot spins in place), check:
```
picarx_dir_motor = [1, 1]   # both normal
```
If one is `-1`, run `1.cali_servo_motor.py` and press `Q` on that motor to flip it back.

---

## Software Stack

```
cam_follower.py / road_follower.py
        │
        ├── picarx (v2.1.0a1)     robot-specific layer
        │       └── robot_hat (v2.5.3)  hardware abstraction
        │               └── lgpio / smbus2  GPIO + I2C
        │
        ├── picamera2 (v0.3.36)   camera capture
        └── OpenCV (v4.10.0)      image processing
```

---

## Stopping the Robot

Always use **Ctrl+C** (not Ctrl+Z) to stop. Ctrl+Z suspends the Python process without running the `finally` block, so the motors keep running.

If the robot is still running after Ctrl+C fails:
```bash
pkill -f cam_follower
pkill -f road_follower
python3 -c "from picarx import Picarx; px=Picarx(); px.stop()"
```

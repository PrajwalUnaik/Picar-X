# Picar-X Autonomous Line Follower

Autonomous lane follower for the **SunFounder Picar-X** robot car, running on a Raspberry Pi. The primary implementation (`lane_follower_v2.py`) uses camera-based PID lane keeping, OpenAI Vision sign detection, and active junction handling. Two simpler reference implementations are also included.

---

## Hardware

| Component | Details |
|---|---|
| Robot | SunFounder Picar-X |
| Platform | Raspberry Pi (aarch64) |
| MCU | STM32-type onboard HAT at I2C 0x14 (PWM / ADC) |
| Sensors | 3× grayscale (A0=Left, A1=Middle, A2=Right) |
| Camera | OV5647, controlled via picamera2 |
| Distance | HC-SR04 ultrasonic (TRIG=D2, ECHO=D3) |
| Camera servos | Pan + Tilt controlled via robot-hat |

---

## Track Environment

Lollipop / keyhole layout: an oval loop connected to a straight via a junction stripe.

```
        ┌──────────────────────────┐
        │     ╔══════════════╗     │
        │     ║  oval loop   ║     │
        │     ╚══════════════╝     │
        │           │ B1           │
        │     ══════╪══════        │  ← junction stripe (GS all-white)
        │           │              │
        │           │ straight     │
        │     ══════╪══════        │  ← junction stripe (GS all-white)
        │           │ B2           │
        └──────────────────────────┘
```

- **Solid white lines** — outer lane boundaries
- **White dashed line** — centre line
- **Junction stripes** — all-white bands detected by grayscale sensors
- **Direction signs** — arrow signs at junctions detected by camera + OpenAI

---

## Setup

### Python dependencies

```bash
pip3 install -r requirements.txt
```

> OpenCV and the `picarx`/`robot-hat` libraries are not on PyPI — they are pre-installed on Raspberry Pi OS and provided as local submodules in this repo respectively.

### OpenAI API key (required for `lane_follower_v2.py`)

`lane_follower_v2.py` uses the OpenAI Vision API (gpt-4o-mini) for sign detection. Create a file at `~/.env` with your key:

```
OPENAI_API_KEY=sk-...your-key-here...
```

The script loads this automatically on startup. Without it, sign detection will fail. Pass `--no-signs` to run lane-following only without the API.

---

## Files

| File | Description |
|---|---|
| `lane_follower_v2.py` | **Recommended** — full autonomous follower with PID, junction handling, OpenAI signs, obstacle recovery |
| `road_follower.py` | Simple grayscale sensor-based line follower |
| `cam_follower.py` | Camera + OpenCV + PID line follower |

---

## `lane_follower_v2.py` — Full Autonomous Follower (Recommended)

### Features

- **PID lane keeping** — HSV white-mask on camera feed, separate left/right blob tracking
- **Junction detection** — all 3 grayscale sensors on white simultaneously = junction stripe
- **Active pan scan** — at junction stripe, car stops, tilts camera up, sweeps ±15°, 3 parallel OpenAI Vision queries, majority vote decides turn direction
- **Background sign polling** — OpenAI Vision queried every ~1s in background thread; arms turn direction before stripe is reached
- **Stop sign handling** — AI arms `stop`, fires on GS stripe, clean exit
- **GS boundary guards** — single sensor on white = lane boundary crossed, hard steer correction
- **Obstacle recovery** — stuck for 2s → back up 3s → pan scan to decide direction → turn and resume
- **MJPEG web stream** — live annotated camera feed at `http://<pi-ip>:8080/stream`
- **Debug photos** — saved to `/tmp/lf_debug/` every 2s and at each junction/scan

### Usage

```bash
python3 lane_follower_v2.py              # follow right lane (default)
python3 lane_follower_v2.py --left-lane  # follow left lane
python3 lane_follower_v2.py --no-signs   # disable OpenAI (lane following only)
python3 lane_follower_v2.py --cam-tilt -20  # override camera tilt angle
```

### Key tuning constants

```python
SPEED_CRUISE  = 12   # % — normal lane-following speed
SPEED_CORRECT = 10   # % — correcting (off-centre)
SPEED_CREEP   =  7   # % — slow creep (gap / junction)
SPEED_TURN    = 11   # % — turning at junction

KP, KI, KD   = 28.0, 0.15, 7.0   # PID gains
MAX_STEER     = 28.0               # degrees max steering
TURN_HOLD_SEC =  2.6               # seconds to hold a junction turn

RECOVERY_BACK_SEC  = 3.0   # seconds to reverse before obstacle recovery scan
JUNCTION_LATCH_SEC = 12.0  # seconds to wait at stripe for AI if scan returns none
AI_POLL_SEC        =  1.0  # background AI query interval
```

### State machine

```
NORMAL → (obstacle 2s) → RECOVERING → (backup 3s + pan scan) → NORMAL
NORMAL → (GS all-white) → [latch] → (scan / wait AI) → TURNING → NORMAL
```

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
python3 road_follower.py
python3 road_follower.py --calibrate
python3 road_follower.py --white-thresh 1400
```

---

## `cam_follower.py` — Camera + OpenCV Follower

Uses the front camera tilted downward to detect the white line with OpenCV, then steers with a PID controller.

### Usage

```bash
python3 cam_follower.py
python3 cam_follower.py --cam-tilt -25
python3 cam_follower.py --save-frames
```

---

## Calibration

### Direction servo (straight-line drift)

```bash
python3 /home/admin/picar-x/example/1.cali_servo_motor.py
```

- Press `1` to select the direction servo
- Press `W`/`S` to adjust offset
- Press `SPACE` then `Y` to save

Calibration stored in `/opt/picar-x/picar-x.conf`.

### Motor direction

If a motor runs backwards, check `picarx_dir_motor = [1, 1]` in the conf file. Run the calibration script and press `Q` on the affected motor to flip it.

---

## Software Stack

```
lane_follower_v2.py
        │
        ├── picarx (v2.1.0a1)       robot-specific layer
        │       └── robot_hat        hardware abstraction (GPIO, I2C, servos)
        │
        ├── picamera2 (v0.3.36)     camera capture
        ├── OpenCV (v4.10.0)        image processing / HSV masking
        ├── openai (v2.41.1)        Vision API for sign detection
        └── flask (v3.1.1)          MJPEG web stream
```

---

## Stopping the Robot

Always use **Ctrl+C** to stop — never Ctrl+Z (suspends without stopping motors).

If the process won't stop:
```bash
pkill -f lane_follower_v2
python3 -c "from picarx import Picarx; px=Picarx(); px.stop()"
```

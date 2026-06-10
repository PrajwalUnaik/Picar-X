# Picar-X Lane Follower — Session Notes

## What is in this branch

| File | Status | Description |
|------|--------|-------------|
| `lane_follower.py` | v1 (stable) | Full two-lane track follower with PID, OpenCV sign detection, state machine |
| `lane_follower_v2.py` | v2 (active dev) | Replaces sign detection with OpenAI Vision API; grayscale junction stripe |
| `cam_follower.py` | Updated v2 | Single dashed-line follower with multi-band scan and angle-aware PID |
| `map.txt` | Reference | ASCII track layout sketch |
| `Image (14).jpg` | Reference | Hand-drawn track diagram (lollipop / keyhole shape) |

---

## Track Layout

Lollipop / keyhole shape:
- **Stem**: straight start/finish section, two lanes (right lane going, left lane returning)
- **Oval**: loop at the top of the stem, joined at junction B
- **Junction B**: where stem meets oval; B2 = right turn into oval, B1 = left turn returning
- **Signs**: directional sign placed on the stem, visible from camera during approach

### Physical marking rules
- 3 tape lines throughout: left boundary (solid), right boundary (solid), centre dashes
- **Junction stripe**: one solid white tape strip (~5cm wide) spanning full lane width,
  placed at the base of the oval (where stem meets oval / where centre dashes end)
- At junction B: outer oval boundary does NOT connect to stem boundaries
- Centre dashes end at the junction stripe; oval has its own dashes forming the loop

---

## lane_follower_v2.py — Architecture

### Design
- **Camera** (Picamera2, 640×480): lane detection via adaptive HSV white mask
- **OpenAI Vision API** (gpt-4o-mini): sign direction, queried every ~2s in background thread
- **Grayscale sensors** (A0/A1/A2): junction stripe detection (all 3 > 600 simultaneously)
- **Ultrasonic**: obstacle stop only (NOT used for junction confirmation on this track)

### State machine
```
NORMAL → TURNING    (junction latch fires when AI responds with direction)
NORMAL → STOP_HELD  (stop sign — not yet used)
TURNING → NORMAL    (after TURN_HOLD_SEC = 2.2s)
```

### Junction latch (key design — added session 2)
The critical timing problem: GS stripe fires, AI responds 1.6s later, robot already past stripe.

Solution: `_junction_seen` flag persists after stripe crossing:
1. GS stripe detected → set `_junction_seen = True`, slow to SPEED_CREEP
2. Every frame: if `_junction_seen AND armed_dir != 'none'` → fire turn immediately
3. `JUNCTION_LATCH_SEC = 6.0` timeout — if AI never responds, clear flag and give up
4. Obstacle stop suppressed when `_junction_seen` is True (expected boundary at junction)

### OpenAI Sign Detector
- `AI_POLL_SEC = 1.0` (effective cycle ~2s including ~1s API latency)
- `AI_ARM_THRESHOLD = 1` (single response arms; stripe is the physical gate)
- Client created once in `start()`, not per-call
- Every response printed to terminal: `[AI]  right  (1.0s)`
- API key stored in `/home/admin/.env`, loaded automatically at startup

### Tuned constants
```python
CAM_TILT           = -15.0   # servo safe range; -38 strained against mechanical stop
LANE_ROI_TOP       = 0.42    # tuned for -15° tilt
LANE_ROI_BOT       = 0.82
GS_WHITE_THRESHOLD = 600     # white tape reads 1400-3600; dark floor reads 200-400
KP, KI, KD        = 22.0, 0.15, 7.0
SPEED_CRUISE       = 14      # % motor speed
TURN_HOLD_SEC      = 2.2
JUNCTION_LATCH_SEC = 6.0
AI_POLL_SEC        = 1.0
AI_ARM_THRESHOLD   = 1
```

---

## lane_follower.py — v1 (what was built in session 1)

### Architecture
- **Camera**: lane detection + OpenCV sign detection (HSV blob analysis)
- **Grayscale**: boundary guard (single sensor alerts)
- **Sign arming**: 20 consecutive frames of sign sighting + lane gap fires the turn
- **Known issue**: OpenCV sign detection noisy in dynamic environments → replaced in v2

### Key constants
```python
SIGN_ARM_FRAMES = 20
GAP_MAX_FRAMES  = 12
SPEED_CRUISE    = 14
TURN_HOLD_SEC   = 2.2
```

---

## cam_follower.py — v2 changes (session 1)

- Resolution: 320×240 → 640×480
- ROI: single band → three bands (near/mid/far) weighted centroid
- PID: plain offset → offset + angle-aware look-ahead (`ANGLE_GAIN`)
- White threshold: fixed → adaptive (tracks mean brightness)
- Speed: binary → confidence-scaled (white pixel count proxy)
- Gap handling: hard switch → decaying bias (`GAP_DECAY`, `GAP_BIAS`)

---

## Debugging history

### Session 1 fixes
| Problem | Root cause | Fix |
|---------|-----------|-----|
| Camera servo straining | CAM_TILT=-38° exceeded -35° limit | Changed to -15° |
| ROI mismatch | Lines at 60-80% but ROI at 5-42% | Moved ROI to 0.42-0.82 |
| TypeError on format string | `right_cx` was None, used `:>5` | Conditional format |
| TFLite disabled silently | Not available on Python 3.13 | Replaced with OpenCV classifier |
| Robot turning immediately | False arm count without lane | Added `last_error_valid` gate |
| Git push auth failure | HTTPS needs token | Token embedded in remote URL |

### Session 2 fixes
| Problem | Root cause | Fix |
|---------|-----------|-----|
| Robot not turning at junction | AI responds after stripe (timing) | Added `_junction_seen` latch |
| Obstacle stop at junction | Hard stop triggered at junction boundary | Suppressed when `_junction_seen` |
| openai not installed | Python 3.13, needed explicit install | `pip3 install openai --break-system-packages` |
| API client overhead | `openai.OpenAI()` created per-call | Moved to `start()`, reused |

---

## What was tried and did NOT work

1. **TFLite sign detection**: not importable on Python 3.13
2. **OpenCV HSV blob**: noisy, false positives from similar-colored objects
3. **Temporal filter + lane-gated arming**: reduced but didn't eliminate false positives
4. **Sonar gate at junction**: this track has no wall directly ahead at B (oval opens up)
5. **One-shot GS stripe trigger**: AI arrived 1.6s after stripe → robot already past it

---

## Known issues / pending

- **Junction latch not yet tested** (session 2 ended before test run)
- `TURN_HOLD_SEC = 2.2` may need tuning depending on turn radius at B
- Camera ROI / white threshold untested on oval section (may need separate tuning)
- Left-lane return path (B1→A) not yet tested; may need `--left-lane` flag

---

## Hardware notes
- Python 3.13.5 — TFLite unavailable, use OpenCV or cloud AI
- Servo tilt range: -35° to +65°; use -15° for front-facing view
- Grayscale: dark mat ≈ 200–400 ADC, white tape ≈ 1400–3600 ADC (confirmed)
- `GS_WHITE_THRESHOLD = 600` confirmed working
- OpenAI API latency on Pi over WiFi: ~1.0–1.8s per call
- `openai` package: installed via `pip3 install openai --break-system-packages`
- API key stored in `/home/admin/.env` (chmod 600), auto-loaded by lane_follower_v2.py

## Dependencies
```
pip3 install openai --break-system-packages
```
OPENAI_API_KEY stored in /home/admin/.env — do NOT commit this file.

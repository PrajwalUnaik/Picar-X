# Picar-X Lane Follower — Session Notes

## What is in this branch

| File | Status | Description |
|------|--------|-------------|
| `lane_follower.py` | New | Full two-lane track follower with PID, sign detection, state machine |
| `cam_follower.py` | Updated v2 | Single dashed-line follower with multi-band scan and angle-aware PID |

---

## lane_follower.py — what was built

### Architecture
- **Camera** (Picamera2, 640×480): primary sensor for lane detection and sign detection
- **Grayscale sensors** (A0/A1/A2): fast boundary-crossing guard, fires before PID can react
- **Ultrasonic**: obstacle stop at configurable distance
- **Background thread**: sign detection runs async so it never blocks the 30 fps drive loop

### State machine
```
NORMAL → STOP_HELD  (stop sign detected)
NORMAL → TURNING    (sign armed + lane gap fires the turn)
TURNING → NORMAL    (after TURN_HOLD_SEC seconds)
STOP_HELD → NORMAL  (after STOP_HOLD_SEC seconds)
```

### Lane detection (`LaneDetector`)
- Adaptive HSV white mask (threshold tracks ambient brightness via `ADAPTIVE_K`)
- ROI: `LANE_ROI_TOP=0.42` to `LANE_ROI_BOT=0.82` (tuned for `CAM_TILT=-15°`)
- Blobs split into left/right halves → weighted centroid per side → lane centre
- If only one boundary visible: other side is inferred from frame edge
- `last_error_valid` flag: True only when lines are actively found

### Sign detection (`SignDetector`)
- **TFLite unavailable on Python 3.13** — silently disabled on every previous run
- Replaced with OpenCV-only classifier (`_opencv_classify`):
  - HSV mask H=0–40 (red/orange/yellow) finds sign blobs
  - For clear signs (bg brightness > 120): classify arrow direction from dark-pixel
    distribution across left/right halves of crop
  - For blurry/dark signs (bg < 120): fall back to sign's horizontal position in
    frame (right half → "right", left half → "left")
- Runs every `SIGN_INTERVAL=0.40s` in background thread

### Sign arming + gap trigger (key design)
The critical insight: **sign detection and turn execution are decoupled**.

1. Sign detector increments `_sign_arm_cnt` each frame the sign is visible,
   **only while lane markings are visible** (`last_error_valid == True`).
   This prevents false arming when the robot is off-track.
2. `_sign_arm_cnt` resets to 0 the moment the sign disappears.
3. A turn is only committed when BOTH:
   - `_sign_arm_cnt >= SIGN_ARM_FRAMES` (20 frames ≈ 0.67s of continuous sighting)
   - Lane markings have ended for ≥ 2 consecutive frames (real junction, not a dash gap)

### Known issues / limitations
- OpenCV sign classifier is still noisy in environments with orange/yellow objects.
  Works in practice because of the `last_error_valid` gate, but false positives exist.
- Arrow direction detection (dark pixel analysis) is unreliable when the sign is small
  or motion-blurred. Falls back to horizontal position heuristic which assumes the sign
  post is on the same side as the intended turn.
- `tflite_runtime` is **not available on Python 3.13**. The vilib install script
  explicitly skips it for Python > 3.12. The TFLite code path is retained in
  `SignDetector` but will never activate on this system.

### Tuned constants (current values)
```python
CAM_TILT        = -15.0   # was -38.0; reduced to stop servo straining against limit
LANE_ROI_TOP    = 0.42    # tuned to match where lines appear at -15° tilt
LANE_ROI_BOT    = 0.82
SIGN_CONFIDENCE = 70      # lowered from 82 (unused — TFLite not available)
SIGN_MIN_AREA   = 900     # lowered from 1600
SIGN_ARM_FRAMES = 20      # frames of continuous sign sighting to arm a turn
KP, KI, KD     = 22.0, 0.15, 7.0
SPEED_CRUISE    = 14
TURN_HOLD_SEC   = 2.2
```

---

## cam_follower.py — what changed (v1 → v2)

- Resolution: 320×240 → 640×480
- ROI: single band → three bands (near/mid/far) with weighted centroid
- PID: plain offset error → offset + look-ahead angle (`ANGLE_GAIN`)
- White threshold: fixed → adaptive (tracks mean frame brightness)
- Speed: binary follow/seek → confidence-scaled (white pixel count proxy)
- Contour filter: none → aspect ratio check rejects non-line blobs
- Gap handling: hard switch → decaying bias (`GAP_DECAY`, `GAP_BIAS`)

---

## What was tried and did NOT work

### ROI placement iterations
The ROI had to be moved three times as the camera tilt changed:
- Original (`-38°` tilt): lines at top 5–30% of frame → ROI 0.05–0.42
- After tilt change to `-15°`: lines dropped to 60–80% of frame → ROI 0.42–0.82

### Sign detection approaches tried
1. **TFLite** (vilib model): not importable on Python 3.13 — silently disabled
2. **Color blob + TFLite**: same problem
3. **HSV blob + white pixel centroid**: sign is too dark/blurry, max gray = 128,
   zero white pixels in the sign crop
4. **HSV blob + dark pixel analysis**: unreliable due to motion blur and similar-
   colored background objects (chairs, equipment, sign pole)
5. **Temporal filter + lane-gated arming**: reduces but does not eliminate
   false positives; still misclassifies the cyan sign pole as a "left" sign

---

## Agreed next architecture (NOT yet implemented)

### Problem statement
The robot navigates a dynamic track with:
- Straight sections (curve handled by PID — no sign needed)
- Dash gaps (brief, no action — hold heading)
- Real junctions (lane ends, robot must turn based on sign)

A reliable junction-navigation system needs two independent signals:
- **Direction**: which way to turn
- **When**: exactly when to execute the turn

### Proposed solution: periodic AI vision + hardware junction detection

**Direction — OpenAI Vision API (async background thread)**
- Send a frame to GPT-4o Vision every 2–3 seconds, unconditionally (not triggered
  by sign detection — this eliminates the pre-detection bootstrapping problem)
- Prompt: *"Is there a junction direction sign visible? Reply with exactly one word:
  left, right, forward, or none."*
- Require 2+ consecutive identical responses before arming the direction
- This handles dynamic layouts, arbitrary sign types, and requires no local ML

**When — Grayscale sensors (hardware)**
- At a real junction, a perpendicular white line (junction marker) triggers all three
  grayscale sensors simultaneously → very reliable hardware detection
- Ultrasonic adds a third gate: obstacle ahead confirms real junction vs. dash gap
- The armed direction fires when grayscale detects junction marker AND ultrasonic
  confirms obstacle ahead

**Why this is better**
- OpenAI is queried while the sign is visible from far away (5–15 seconds of
  approach time at slow speed) — latency is not a problem
- The AI is never consulted at the junction itself (where camera can't see side paths)
- `last_error_valid` gate is replaced by the grayscale hardware trigger — much more
  reliable than software lane detection
- False positive AI responses are filtered by requiring consecutive agreement
- Dynamic layout works because the AI reads whatever sign is actually in the scene

### Implementation plan (when ready to build)
1. Add `OpenAISignDetector` class (replaces `SignDetector`) with async thread
2. Add junction detection using all-three-sensors-on-white pattern in `_gs_alert()`
3. Arm direction from AI response (2+ consecutive agreement required)
4. Fire turn when junction marker AND ultrasonic obstacle both trigger
5. After turn, reset arm state and re-acquire lane via PID

### Dependencies needed
- `openai` Python package (`pip install openai`)
- OpenAI API key (set as env var `OPENAI_API_KEY` or in `.env`)
- Stable WiFi on the Pi (already present)

---

## Hardware notes (this robot)
- Python 3.13.5 — TFLite unavailable, use OpenCV or cloud AI
- Servo tilt range: -35° to +65°; -38° strains against mechanical stop → use -15°
- Grayscale calibration: dark mat ≈ 200–400 ADC, white tape ≈ 800–2000 ADC
- `GS_WHITE_THRESHOLD = 600` is the current boundary detection threshold
- Motor calibration: `picarx_dir_motor = [1, 1]` (no direction flip needed)

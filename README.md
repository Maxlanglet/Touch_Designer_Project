# Hands

Real-time hand-gesture-driven video effects. Point your webcam at yourself, make a
gesture, and watch visual effects get applied to your hands in real time. Gestures are
recognized with [MediaPipe](https://ai.google.dev/edge/mediapipe), effects are rendered
with [OpenCV](https://opencv.org/), and everything is wrapped in a live
[PySide6](https://doc.qt.io/qtforpython-6/) GUI with tunable controls.

## Example

<!-- Add an example screenshot/GIF here -->
<!-- ![Hands demo](docs/example.png) -->

_Coming soon._

## Features

- **Live gesture recognition** from the webcam (up to two hands) via MediaPipe's
  gesture recognizer.
- **Gesture → effect bindings**: each gesture maps to a pool of effects that are applied
  to regions tracked on your hands.
- **Built-in effects**: Canny edges, Sobel edges, pixelate, thermal, and blur.
- **Built-in gestures**: two-hand pinch-squeeze, closed fists, and thumbs-up (used to
  reset the active effects).
- **Config-driven**: gestures, effects, and their parameters are declared in
  `src/config/app.json` — no code changes needed to re-map things.
- **Live control panel**: tweak gesture and effect parameters on the fly and watch the
  FPS counter, all from the side panel in the GUI.
- **Pluggable architecture**: add new gestures or effects by writing a class and
  registering it with a decorator.

## Requirements

- Python `>=3.11, <3.15`
- A working webcam
- [Poetry](https://python-poetry.org/) for dependency management

Core dependencies (installed automatically by Poetry):

- `opencv-python` — frame capture and image processing
- `mediapipe` — hand landmark and gesture recognition
- `pyside6` — the desktop GUI
- `pydantic` — config models
- `loguru` — logging

## Installation

```bash
# Clone the repo
git clone <your-repo-url> hands
cd hands

# Install dependencies into a virtual environment
poetry install
```

The MediaPipe model files are included under `models/`
(`gesture_recognizer.task` and `hand_landmarker.task`), so no extra downloads are
required.

## Usage

Run the app from the project root (the model and config paths are relative to it):

```bash
poetry run python -m src.main
```

Then:

1. Allow camera access if prompted.
2. Use the side **control panel** to adjust gesture/effect parameters live.
3. Make a bound gesture (e.g. **closed fists** or a **two-hand pinch-squeeze**) to apply
   effects.
4. Give a **thumbs-up** to reset/clear the active effects.
5. Close the window to quit.

### Quick standalone demo

`test.py` is a minimal, self-contained MediaPipe + OpenCV demo that draws hand landmarks
and the top recognized gesture in a plain OpenCV window. Useful for sanity-checking your
camera and the model:

```bash
poetry run python test.py
# press "q" to quit
```

## Configuration

All gesture/effect wiring lives in `src/config/app.json`. It has two sections:

- **`bindings`** — maps each gesture to an ordered list of effects (with per-effect
  parameters). Effects in a pool are assigned to the regions a gesture produces.
- **`gesture_params`** — tuning parameters for each gesture detector (thresholds,
  hold/grace timing, smoothing).

Example binding:

```json
{
  "gesture": "closed_fists",
  "effects": [
    { "name": "canny_edges", "params": { "render_outline": true, "threshold1": 50, "threshold2": 50 } },
    { "name": "pixelate",    "params": { "render_outline": true, "block_size": 30 } }
  ]
}
```

Changes you make in the GUI can be persisted back to a config file via the
`save_config` helper in `src/config/config.py`.

## Architecture

The app is a pipeline that turns each camera frame into a rendered output:

```
Webcam (OpenCV)
   │  frame
   ▼
Detector            ─ runs MediaPipe, extracts hand landmarks
   │  hands
   ▼
GestureManager      ─ runs each registered GestureDetector
   │  detected gestures
   ▼
EffectManager       ─ maps gestures → effect pools, tracks active regions
   │
   ▼
Renderer / Effects  ─ draws landmarks + applies effects to regions
   │  processed frame
   ▼
PySide6 GUI (VideoWidget + ControlPanel)
```

Key modules:

| Path | Responsibility |
|------|----------------|
| `src/main.py` | Wires everything together and launches the Qt app. |
| `src/service/detector.py` | Wraps the MediaPipe gesture recognizer; produces `HandData`. |
| `src/service/frame_pipeline.py` | Orchestrates detect → handle → render per frame. |
| `src/gestures/` | Gesture detectors + `GestureManager` and the gesture registry. |
| `src/effects/` | Effects + `EffectManager` and the effect registry. |
| `src/renderer/` | Drawing primitives for hands, gestures, and effect regions. |
| `src/config/` | Config models and the factory that builds managers from JSON. |
| `src/ui/` | PySide6 main window, video widget, control panel, and worker thread. |
| `src/geometry/`, `src/models/` | Normalized geometry helpers and hand data models. |

## Extending

### Add a new effect

1. Create a class in `src/effects/` that subclasses `Effect` and implements `apply(...)`.
2. Decorate it with `@register_effect("my_effect")`.
3. Reference `"my_effect"` in `src/config/app.json`.

### Add a new gesture

1. Create a class in `src/gestures/` that subclasses `GestureDetector` and implements
   `detect(...)` and `reset(...)`.
2. Decorate it with `@register_gesture("my_gesture")` and add a matching `GestureType`.
3. Bind it to effects in `src/config/app.json`.

## Development

This project uses [pre-commit](https://pre-commit.com/) with
[Ruff](https://docs.astral.sh/ruff/) for fast linting and formatting:

```bash
poetry add --group dev pre-commit ruff   # if not already installed
poetry run pre-commit install            # enable git hook
poetry run pre-commit run --all-files    # run on the whole repo
```

## Roadmap

See `TODO.md` for planned work, including additional gestures (palm touching, single-hand
pinch), more effects (noise, ASCII, glitch, frequency-based), and performance work.

## License

No license specified yet.

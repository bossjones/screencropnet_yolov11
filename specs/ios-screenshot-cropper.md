# Spec: iOS Screenshot Cropper

A local-only iOS app that auto-detects the tweet region in a Twitter/X screenshot,
lets you adjust the crop and pick an aspect ratio, and batch-saves the results back
to Photos — replacing the manual InShot workflow for this one task.

> **Status:** design spec. No code, Xcode project, or model export exists yet. This
> document is the blueprint for a future TDD implementation.

## Context — why this exists

Screenshots of tweets are currently cropped/resized by hand in InShot. This repo already
contains a model (`screencropnet_yolo`) that locates the tweet region in a screenshot.
The goal is a small SwiftUI app that uses that model on-device to do the tedious part —
finding and framing the crop — while keeping a human in the loop to nudge each result.

### Key reframing discovered during exploration

- **The model is a *detection* model, not segmentation.** `src/screencropnet_yolo/model.py`
  loads YOLO26 detection weights; `config/config.yaml` and
  `datasets/twitter_screenshots_localization_dataset/data.yaml` define a single class
  `tweet_region`. `src/screencropnet_yolo/inference.py` returns
  `Detection(bbox=(x1,y1,x2,y2), confidence, ...)` — axis-aligned boxes, no polygons or
  masks. A crop is a rectangle, so a bounding box is exactly the right primitive.
  Wherever "segmentation coordinates" was assumed, read "bounding box."
- **CoreML export already exists.** `ModelExporter` in `model.py` lists `"coreml"` in
  `SUPPORTED_FORMATS`, so on-device inference is viable without new training work.
- **The repo's own server is not used by this app.** `src/screencropnet_yolo/server/api.py`
  (`POST /classify`) is a heavy async RabbitMQ + Postgres job queue. The separate
  `fastapi_pytorch_postgresql_sandbox` repo serves a *different* model (an EfficientNet
  platform classifier: facebook/tiktok/twitter) and is irrelevant here.

## Locked decisions

| Decision | Choice |
|---|---|
| Inference location | **On-device CoreML** — export once, bundle, run via Vision |
| Per-image interaction | **Auto-suggest, then adjust** — model proposes the crop, user nudges, confirms |
| Aspect-ratio behavior | **Expand box to target ratio** — grow outward, never clip the tweet |
| Input / output | **Photos in, Photos out** — pick from Photos, save to an album |
| Project structure | **SPM core package + thin SwiftUI app** — fast `swift test` TDD loop |
| Minimum iOS | **iOS 17+** — modern SwiftUI, Vision, Swift Testing |
| Aspect presets | **1:1, 4:5, 9:16, Original/freeform** |

## Objective

Implement a SwiftUI iOS 17+ app that imports screenshots from Photos, runs the bundled
CoreML screencropnet model on-device to suggest a tweet-region crop, lets the user adjust
the box and pick an aspect ratio (which expands the crop to that ratio without clipping),
and batch-saves the results to a Photos album — all testable locally in the Simulator,
with the core logic testable without a simulator at all.

---

## Architecture

Two layers, chosen specifically so an AI agent (or a human) can drive a tight TDD loop:

```
ScreenshotCropper/                 (Xcode workspace)
├── SwiftCropCore/                 (Swift Package — NO UIKit/Photos/CoreML imports)
│   ├── Sources/SwiftCropCore/
│   │   ├── CropRect.swift         normalized rect, top-left origin, 0…1
│   │   ├── AspectRatio.swift      enum: square / portrait45 / story916 / original / custom(w,h)
│   │   ├── CropEngine.swift       pure: detection + imageSize + ratio → CropPlan (pixels)
│   │   ├── RegionDetector.swift   protocol: detect(in:) async throws -> [Detection]
│   │   ├── MockDetector.swift     returns fixed Detections for tests
│   │   └── BatchPlanner.swift     orchestrates many images → many CropPlans
│   └── Tests/SwiftCropCoreTests/  Swift Testing — run with `swift test`, no simulator
└── ScreenshotCropperApp/          (thin Xcode app target, iOS 17+)
    ├── CoreMLRegionDetector.swift  Vision/CoreML impl of RegionDetector
    ├── PhotosLibraryService.swift  PHPhotoLibrary import + album save
    ├── CropImageRenderer.swift     applies a CropPlan (CGImage crop + optional pad)
    ├── Views/ (SwiftUI)            grid, single-image adjust w/ draggable box, batch review
    ├── ScreencropnetDetector.mlpackage   bundled CoreML model
    └── UITests/                    XCUITest — run via `xcodebuild test` in Simulator
```

**The boundary is the whole point:** every decision with logic (coordinate conversion,
ratio expansion, clamping/padding, batch sequencing) lives in `SwiftCropCore`, depends
only on Foundation / CoreGraphics types (`CGSize`, `CGRect`, numbers), and is tested by
`swift test` in seconds with no image and no simulator. The app target only does I/O and
pixels.

### Core data types & the central pure function

- `CropRect`: normalized `(x, y, width, height)`, **top-left origin**, values 0…1. The
  app's single coordinate convention.
- `Detection`: `CropRect` + `confidence`. Detector implementations must convert into this
  convention.
- `AspectRatio`: `.square` (1:1), `.portrait` (4:5), `.story` (9:16), `.original`,
  `.custom(w, h)`.
- `CropPlan`: a pixel-space `CGRect` to cut, plus optional `padding` (insets + fill) for
  when the image can't supply enough pixels to hit the ratio.

The heart of the app, fully unit-testable:

```swift
CropEngine.plan(for detection: Detection,
                imageSize: CGSize,      // pixels
                ratio: AspectRatio) -> CropPlan
```

Algorithm (expand, never clip):

1. Convert `detection.rect` (normalized) → pixel `CGRect` using `imageSize`.
2. If `ratio == .original`, return that rect as the plan.
3. Compute target ratio `r = targetW / targetH`. Current box ratio `b = w / h`.
   - If `b < r` (too narrow): new width = `h * r`; grow width symmetrically about center.
   - If `b > r` (too short): new height = `w / r`; grow height symmetrically about center.
4. Clamp the grown rect to `imageSize`. If clamping would break the ratio (the box is near
   an edge), first try shifting the box along that axis to stay inside.
5. If the image still cannot supply the exact ratio (the box already spans the full image
   on the constraining axis), record `padding` so the renderer pads with a configurable
   fill (white / black / blurred-extend) to guarantee the exact output ratio. This
   preserves the "never cut off the tweet" promise.

Edge cases the tests must cover: zero detections; detection touching an image edge;
detection already wider/taller than the target ratio; portrait vs landscape source; ratio
that exceeds the image on both axes; degenerate 1px boxes.

### Detector protocol (enables both TDD and on-device inference)

```swift
protocol RegionDetector {
    func detect(in image: CropImage) async throws -> [Detection]   // sorted by confidence desc
}
```

- `MockDetector` (in core, no dependencies): returns fixed `[Detection]` — used by every
  core test and by SwiftUI previews.
- `CoreMLRegionDetector` (app target): wraps `VNCoreMLRequest` over the bundled
  `.mlpackage`. Converts Vision's **bottom-left** normalized `VNRecognizedObjectObservation`
  boxes into the core's **top-left** `CropRect`, handling EXIF / `CGImagePropertyOrientation`
  from screenshots.

### Multiple detections

Screenshots usually contain one tweet region. The default suggestion is the
highest-confidence box above a threshold (e.g. `0.5`). If more than one box clears the
threshold, the adjust screen lets the user tap to choose which box seeds the crop. If no
box clears the threshold, the image is flagged in the review list for full-manual cropping.

### App flow (SwiftUI)

1. **Import** — `PHPicker` / Photos grid, multi-select of screenshots.
2. **Per-image adjust** — show the suggested crop as a draggable/resizable overlay; an
   aspect-ratio control (1:1 / 4:5 / 9:16 / Original); changing the ratio re-runs
   `CropEngine.plan`; live preview.
3. **Batch review** — thumbnail grid of all planned crops; per-item confidence/flag badges;
   "Save all."
4. **Save** — `CropImageRenderer` produces each cropped image; `PhotosLibraryService` writes
   them to a dedicated album (e.g. `ScreenCrop`).

---

## The TDD / agent feedback loop

Three concentric loops, fastest first — this is how the build is steered with TDD:

1. **Core loop (seconds, no simulator):** `swift test` against `SwiftCropCore`. All crop
   math, ratio expansion, clamping/padding, coordinate conversion, and batch sequencing are
   pure functions tested with `MockDetector` and plain `CGSize` / `CGRect` values. Roughly
   90% of the logic is driven red→green→refactor here. This is where TDD actually shapes the
   design.
2. **App loop (tens of seconds, Simulator):**
   `xcodebuild test -scheme ScreenshotCropperApp -destination 'platform=iOS Simulator,name=iPhone 15'`.
   XCUITest drives the SwiftUI flow end-to-end; optional **snapshot tests**
   (`pointfreeco/swift-snapshot-testing`) catch visual regressions in the crop overlay and
   rendered output. CoreML/Vision runs in the Simulator, so even on-device inference is
   exercised locally.
3. **Device loop (manual, occasional):** only when you want it on a physical iPhone.
   **TestFlight is NOT required for development** — it is a distribution channel. A tethered
   device can also be run from Xcode with a free Apple ID. Reserve TestFlight for sharing
   builds or device-only validation.

**Simulator vs TestFlight, explicitly:** the Simulator covers building, unit tests, UI
tests, screenshots, Photos access, and CoreML inference. TestFlight (or direct device
install) is only needed to confirm real-device camera-roll behavior, measure performance on
actual hardware, or hand the app to someone else.

---

## Tooling / plugins / setup checklist

**Required:**

- **Xcode 16+** (App Store) — provides the iOS 17 SDK, Swift Testing, and Simulators.
- Command-line tools: `xcode-select --install`, then `sudo xcodebuild -license accept`.
- Verify the toolchain before any code:
  - `xcodebuild -version`
  - `swift --version`
  - `xcrun simctl list devices` (confirm an iPhone Simulator exists)

**Recommended for a clean agent loop:**

- `brew install xcbeautify` — human/agent-readable `xcodebuild` output (pipe builds through
  it).
- `brew install xcodegen` — generate the app target's Xcode project from a YAML manifest so
  no one hand-edits an unmergeable `.pbxproj`. (`SwiftCropCore` is pure SPM and needs no
  generator.)
- `brew install swiftlint` — lint/style gate.
- SPM dev dependency `pointfreeco/swift-snapshot-testing` — visual-regression tests for the
  crop UI/output.

**For the CoreML export (done once, in *this* Python repo, not the iOS app):**

- Export via the existing `ModelExporter` / Ultralytics:
  `model.export(format="coreml", nms=True)` — `nms=True` embeds non-max-suppression so
  Vision returns clean `VNRecognizedObjectObservation`s. Produces a `.mlpackage`.
- Needs `coremltools` in the environment (Ultralytics pulls it; if missing, `uv add
  coremltools`). Run through `uv` per repo convention.
- Copy the resulting `.mlpackage` into the app target and confirm its class label maps to
  `tweet_region`.

No Claude Code "plugin" is needed for Swift specifically — the agent drives everything
through `swift`, `xcodebuild`, and `xcrun simctl`. The brew tools above are the practical
extras.

---

## Testing strategy

- **Unit (core, primary):** Swift Testing suites for every `CropEngine` branch and edge
  case listed above; `BatchPlanner` ordering and flagging; coordinate-convention
  round-trips. Target near-100% coverage on `SwiftCropCore`.
- **Integration (app):** `CoreMLRegionDetector` against a couple of fixture screenshots
  (assert a box is found, confidence is sane, orientation is correct);
  `PhotosLibraryService` against the Simulator's Photos.
- **UI (app):** XCUITest for the import → adjust → batch-save happy path; snapshot tests for
  the overlay and rendered crops at each preset.

## Implementation outline (future work — not part of this spec)

1. Scaffold `SwiftCropCore`; TDD `CropRect`, `AspectRatio`, `CropEngine`, `BatchPlanner`,
   `MockDetector`.
2. Export the CoreML model from this repo; bundle the `.mlpackage` into the app.
3. Build the thin app target (XcodeGen) with `CoreMLRegionDetector`, `PhotosLibraryService`,
   `CropImageRenderer`, and the SwiftUI views.
4. Add XCUITest + snapshot tests.

## Acceptance criteria (for the eventual implementation)

- `swift test` passes for `SwiftCropCore` with the edge cases above covered.
- The app imports screenshots from Photos, shows a model-suggested crop, supports adjust +
  the four aspect presets with expand-to-ratio behavior, and saves a batch to a Photos
  album.
- Inference runs on-device via the bundled CoreML model in the Simulator (no network, no
  server).

## Open follow-ups (settle at implementation time)

- Padding-fill default (white vs blurred-extend).
- Confidence threshold value.
- Album name.
- Whether to add an optional fully-automatic batch mode on top of the review flow.

## Notes

- The Photos save path requires the `NSPhotoLibraryAddUsageDescription` (add-only) key, and
  reading requires `NSPhotoLibraryUsageDescription`, in the app's Info.plist.
- Vision normalizes coordinates with a **bottom-left** origin; the core uses **top-left**.
  The conversion (and screenshot EXIF orientation) is the most error-prone seam — cover it
  with explicit round-trip tests.

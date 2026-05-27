---
paths: tests/**/*.py, **/test_*.py, **/conftest.py
---

# Visual & Image Testing Conventions

Plots, images, and tensors don't fit `assertEqual` — compare them by numeric or
perceptual tolerance, never byte-for-byte. These rules extend (don't replace)
`ai_docs/pytorch-testing-notes.md`; the full pattern catalog with copy-paste
code lives in `ai_docs/html/visual-testing-patterns.html`.

When a test produces a plot, image, or tensor:

1. Test the data, not the pixels, first - Interrogate the matplotlib `Axes`
   (`ax.patches` heights, `line.get_xdata()`, tick labels) with
   `np.testing.assert_array_equal`. Most plot bugs are math bugs in disguise.
2. Use pixel/baseline tests only for "does it look right" - Reach for
   `pytest-mpl` (`@pytest.mark.mpl_image_compare`) with an explicit
   `tolerance=`, commit baselines under `tests/baseline/`, and force the Agg
   backend. Never generate baselines on macOS for Linux CI.
3. Pick the image-diff tool by intent - `cv2.absdiff().mean() < tol` for exact
   output from the same pipeline; `skimage`'s `structural_similarity`
   (SSIM ≥ ~0.95) for anything lossy (JPEG/resize); `imagehash.phash` Hamming
   distance ≤ 5 for format/size-agnostic "same picture". Never assert exact
   equality on a lossy image.
4. Keep visual tests deterministic and offline - Rely on the autouse
   `_deterministic` fixture (`tests/conftest.py`, seeds `torch.manual_seed(0)`),
   set `matplotlib.use("Agg")`, and `mocker.patch` disk/network/model I/O. No
   dataset downloads, no training loops.
5. Make failures inspectable - On mismatch, write `actual.png` and an amplified
   `diff.png` (×10) into `tmp_path`; gate baseline overwrites behind a
   deliberate `--update-baselines` option, never silently.
6. Compare tensors with tolerance - Use
   `torch.testing.assert_close(actual, expected, rtol=…, atol=…)`; exact `==`
   only for shapes/ints. `pytest-pytorch` is for contributing to PyTorch
   itself, not for testing code that merely uses it.

Visual-testing tools (`pytest-mpl`, `scikit-image`, `imagehash`,
`opencv-python`) are not yet project dependencies. Add them only when a test
first needs one, via `uv add --dev <pkg>` — never `uv pip install`.

## Correct

```python
def test_revenue_bars_match_input():
    """Math-first: assert the plotted heights, not the pixels."""
    fig, ax = plt.subplots()
    ax.bar(["q1", "q2", "q3"], [1.2, 1.8, 2.6])
    heights = [b.get_height() for b in ax.patches]
    assert heights == [1.2, 1.8, 2.6]


def test_thumbnail_perceptually_matches():
    """Lossy output → SSIM with an explicit threshold, not byte equality."""
    score, _ = ssim(expected_gray, actual_gray, full=True)
    assert score >= 0.95, f"SSIM {score:.3f} below 0.95"
```

## BAD

```python
def test_chart():
    fig = make_chart()
    assert fig == load_baseline()          # figures aren't ==-comparable


def test_thumbnail():
    actual = Image.open("build/out.jpg").tobytes()
    expected = Image.open("tests/fixtures/out.jpg").tobytes()
    assert actual == expected              # exact bytes on a lossy JPEG: flaky
```

## Why This Matters

Byte-for-byte assertions on visual output fail for reasons that have nothing to
do with correctness — font stacks, anti-aliasing, JPEG round-trips, matplotlib
versions. They produce flaky CI and train the team to ignore failures.
Tolerance- and data-based assertions test what the code actually promises, stay
stable across machines, and turn a red build into a real question: *is the new
image correct?*

**Remember:** Assert the math when you can, perceptual tolerance when you must,
exact pixels only when the pipeline is deterministic.

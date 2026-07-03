# Scalene profile — `tests/test_inference_e2e.py`

- Date: 2026-07-02
- Target: `tests/test_inference_e2e.py`
- Backend: Scalene (line-level CPU / native / memory / GPU)

## Command

```bash
uv run --with scalene python -m scalene run -o /var/folders/wg/9dy_d45j0cbdj9fg2xcvxyvm0000gn/T/tmp391kf6sn/scalene.json /Users/bossjones/dev/bossjones/screencropnet_yolov11/.claude/skills/perf/scripts/_pytest_runner.py --- tests/test_inference_e2e.py -p no:cov -o addopts= --timeout=0 --import-mode=importlib -m integration
```

## Hottest lines

| File | Line | CPU % | Python % | Native % | Peak MB | GPU % | Code |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/.claude/skills/perf/scripts/_pytest_runner.py` | 15 | 89.4 | 0.0 | 89.4 | 31151.3 | 0.0 | `raise SystemExit(pytest.main(sys.argv[1:]))` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/.claude/skills/perf/scripts/_pytest_runner.py` | 12 | 0.9 | 0.2 | 0.7 | 14.1 | 0.0 | `import pytest` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/inference.py` | 21 | 0.0 | 0.0 | 0.0 | 6.0 | 0.0 | `import cv2` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/inference.py` | 23 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | `import numpy.typing as npt` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/inference.py` | 24 | 0.0 | 0.0 | 0.0 | 19939.5 | 0.0 | `from ultralytics import YOLO` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/inference.py` | 134 | 0.0 | 0.0 | 0.0 | 44.7 | 0.0 | `self.model = YOLO(model_path, task="detect")` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/inference.py` | 169 | 0.0 | 0.0 | 0.0 | 3.1 | 0.0 | `img = cv2.imread(image)` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/inference.py` | 179 | 0.0 | 0.0 | 0.0 | 19147.6 | 0.0 | `results = self.model.predict(` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/compression.py` | 12 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | `from PIL import Image` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/config.py` | 13 | 0.0 | 0.0 | 0.0 | 10.0 | 0.0 | `from pydantic_settings import BaseSettings, SettingsConfigD…` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/config.py` | 16 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | `class Settings(BaseSettings):` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/db.py` | 23 | 0.0 | 0.0 | 0.0 | 3.0 | 0.0 | `from screencropnet_yolo.server.schemas import StatusSummary` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/metrics.py` | 10 | 0.0 | 0.0 | 0.0 | 13.0 | 0.0 | `from prometheus_client import Counter, Gauge, Histogram, ma…` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/queue.py` | 12 | 0.0 | 0.0 | 0.0 | 32.6 | 0.0 | `import aio_pika` |
| `/Users/bossjones/dev/bossjones/screencropnet_yolov11/src/screencropnet_yolo/server/schemas.py` | 37 | 0.0 | 0.0 | 0.0 | 3.0 | 0.0 | `class QueueMessage(BaseModel):` |

## Optimization candidates

Read the hottest rows above against `src/screencropnet_yolo/inference.py` (`InferencePipeline.predict_image/predict_batch/predict_video`) and `model.py`. Cross-check the numbers with `evaluation.py::benchmark_model`. List concrete, testable changes here before touching code.

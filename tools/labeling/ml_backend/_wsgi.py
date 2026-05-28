"""WSGI entry point for the Label Studio ML backend.

Standard launcher used by ``label-studio-ml start``; also runnable directly with
``python _wsgi.py`` for local debugging.
"""

from __future__ import annotations

import os

from label_studio_ml.api import init_app

from model import TweetRegionModel

app = init_app(model_class=TweetRegionModel)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "9090")), debug=True)  # noqa: S104

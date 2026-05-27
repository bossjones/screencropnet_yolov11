#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# ///

import fcntl
import json
import os
import sys
from pathlib import Path

def main():
    try:
        # Read JSON input from stdin
        input_data = json.load(sys.stdin)

        # Ensure log directory exists
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'post_tool_use.json'

        # Parallel agents fire PostToolUse concurrently; an unlocked
        # read-modify-write corrupts the JSON array. Hold an exclusive flock
        # across the whole transaction. 'a+' creates the file if missing.
        with open(log_path, 'a+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
            if not isinstance(log_data, list):
                log_data = []
            log_data.append(input_data)
            f.seek(0)
            f.truncate()
            json.dump(log_data, f, indent=2)

        sys.exit(0)

    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Exit cleanly on any other error
        sys.exit(0)

if __name__ == '__main__':
    main()

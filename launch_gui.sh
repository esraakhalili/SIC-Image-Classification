#!/bin/bash
# Launcher script for YOLOv8 Classification Suite GUI

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 gui_app.py "$@"


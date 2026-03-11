#!/bin/bash
PID_FILE="$(dirname "$0")/../logs/phantom.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    kill "$PID" 2>/dev/null && echo "👻 Phantom stopped (PID $PID)"
    rm "$PID_FILE"
else
    echo "No daemon running."
fi

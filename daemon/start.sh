#!/bin/bash
# Lance le daemon Phantom en arrière-plan
cd "$(dirname "$0")/.."
nohup python3 daemon/collector.py >> logs/daemon.log 2>&1 &
echo "👻 Phantom daemon started (PID $!)"
echo $! > logs/phantom.pid

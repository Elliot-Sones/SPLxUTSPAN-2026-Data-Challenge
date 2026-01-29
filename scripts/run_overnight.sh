#!/bin/bash
# Run overnight experiments in background, preventing sleep and surviving interruptions.
#
# Usage:
#   ./scripts/run_overnight.sh           # Full run
#   ./scripts/run_overnight.sh --validate # Quick validation only
#
# Monitor:
#   tail -f output/overnight/overnight_search.log
#
# This script uses caffeinate to prevent sleep on Mac.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/output/overnight"
LOG_FILE="$OUTPUT_DIR/overnight_search.log"
STDOUT_FILE="$OUTPUT_DIR/overnight_stdout.log"
PID_FILE="$OUTPUT_DIR/overnight.pid"

mkdir -p "$OUTPUT_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Overnight experiments already running (PID: $OLD_PID)"
        echo "To monitor: tail -f $LOG_FILE"
        echo "To stop: kill $OLD_PID"
        exit 1
    else
        rm "$PID_FILE"
    fi
fi

echo "Starting overnight experiments..."
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Run with caffeinate to prevent sleep (-i: prevent idle sleep, -s: prevent system sleep)
# nohup ensures the process survives terminal close
# The --resume flag means it will pick up where it left off if interrupted

cd "$PROJECT_DIR"

if [[ "$1" == "--validate" ]]; then
    echo "Running validation only..."
    caffeinate -i -s nohup uv run python scripts/overnight_experiments.py --validate > "$STDOUT_FILE" 2>&1 &
else
    echo "Running full overnight experiments..."
    caffeinate -i -s nohup uv run python scripts/overnight_experiments.py --resume > "$STDOUT_FILE" 2>&1 &
fi

PID=$!
echo $PID > "$PID_FILE"

echo ""
echo "Started background process with PID: $PID"
echo ""
echo "Commands:"
echo "  Monitor:  tail -f $LOG_FILE"
echo "  Stdout:   tail -f $STDOUT_FILE"
echo "  Status:   ps -p $PID"
echo "  Stop:     kill $PID"
echo ""
echo "The process will continue even if:"
echo "  - You close the terminal"
echo "  - Your computer would normally sleep (caffeinate prevents this)"
echo ""
echo "Results will be saved to: $OUTPUT_DIR/overnight_results.csv"

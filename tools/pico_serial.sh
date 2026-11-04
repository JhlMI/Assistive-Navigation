#!/bin/bash
# Opens the Pico serial monitor (exit with Ctrl+])
if pgrep -f "python main.py" > /dev/null; then
    echo "main.py is running and holds the port. Stop it first: ~/stop_test.sh"
    exit 1
fi
PORT=$(ls /dev/ttyACM* 2>/dev/null | head -1)
if [ -z "$PORT" ]; then
    echo "Pico not found (/dev/ttyACM*). Check the USB cable."
    exit 1
fi
source ~/yolo_env/bin/activate
echo "Pico on $PORT  |  T,0 = no timeout  |  F L R X S P  |  exit: Ctrl+]"
python -m serial.tools.miniterm "$PORT" 115200

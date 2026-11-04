#!/bin/bash
# Demo: replays a street video in a loop; the models and the motors run live.
# Usage: ~/demo.sh [video.avi] [real_fps for videos recorded before v6_demo]
VIDEO=${1:-$(ls -t ~/Assistive-Navigation/logs/navigation/rec_*.avi | head -1)}
FPS_ARG=""
[ -n "$2" ] && FPS_ARG="--video-fps $2"
sudo nvpmodel -m 0
cd ~/Assistive-Navigation/harness_app
source ~/yolo_env/bin/activate
nohup python main.py --video "$VIDEO" $FPS_ARG --loop > ~/demo_run.log 2>&1 &
sleep 15
head -8 ~/demo_run.log
IP=$(hostname -I | awk '{print $1}')
echo "Demo screen (evaluator): http://$IP:8080/clean"
echo "Control (you):           http://$IP:8080"

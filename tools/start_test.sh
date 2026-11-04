#!/bin/bash
sudo nvpmodel -m 0
cd ~/Assistive-Navigation/harness_app
source ~/yolo_env/bin/activate
export DEPTHAI_DISABLE_CRASHDUMP_COLLECTION=1
nohup python main.py > ~/street_run.log 2>&1 &
sleep 20
head -8 ~/street_run.log
echo "Web: http://$(hostname -I | awk '{print $1}'):8080"

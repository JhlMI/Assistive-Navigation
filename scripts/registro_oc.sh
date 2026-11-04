#!/bin/bash
LOG=~/Assistive-Navigation/logs/registro_oc_$(date +%Y%m%d_%H%M%S).log
echo "Registrando en $LOG (Ctrl+C para parar)"
while true; do
    OC=$(cat /sys/class/hwmon/hwmon3/oc*_event_cnt | tr '\n' ' ')
    TG=$(timeout 1 tegrastats --interval 400 | head -1 | grep -o "VDD_IN [0-9]*mW\|tj@[0-9.]*C" | tr '\n' ' ')
    echo "$(date +%H:%M:%S.%N | cut -c1-12) oc: $OC| $TG" >> "$LOG"
    sync "$LOG"
    sleep 0.5
done

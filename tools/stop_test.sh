#!/bin/bash
pkill -INT -f "python main.py"
sleep 4
tail -6 ~/street_run.log

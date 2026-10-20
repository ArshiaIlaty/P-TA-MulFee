#!/bin/bash
# Self-looping DISCO-TAB watchdog (crond unavailable). Checks every 10 min and
# notifies on finish/crash/idle via watchdog_check.sh. Launch detached:
#   setsid nohup ./watchdog.sh >/dev/null 2>&1 < /dev/null &
cd /home/ec2-user/P-TA-MulFee/disco
mkdir -p logs
while true; do
  ./watchdog_check.sh >> logs/watchdog.log 2>&1
  sleep 600
done

#!/usr/bin/env bash
# Simulazione ROS 2 senza finestre di un caso preparato da demo_ros_prep.py.
#   bash run_case.sh lung_001
set -e
CID="$1"
PROJ="/mnt/c/Users/ilgra/Documents/Progetto Medical"
OUT="$PROJ/outputs/eval/demo_finetune_all/ros"
source /opt/ros/humble/setup.bash
bash "$PROJ/ros2_ws/build.sh" > "$OUT/${CID}_build.log" 2>&1
source "$HOME/ck_ws/install/setup.bash"
rm -f "$OUT/${CID}_resoconto.json"
timeout 1500 ros2 launch cyberknife_ros cyberknife.launch.py \
  rviz:=false bev:=false tumore:=false final_hold_s:=0.0 \
  report_path:="$OUT/${CID}_resoconto.json" > "$OUT/${CID}_ros.log" 2>&1 || true
ls -la "$OUT/${CID}_resoconto.json"

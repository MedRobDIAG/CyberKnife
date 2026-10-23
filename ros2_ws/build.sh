#!/usr/bin/env bash
# ============================================================================
#  Compila il pacchetto ROS 2. Va lanciato DA WSL, non da Windows.
#
#  Perche' build e install finiscono in ~/ck_ws e non qui accanto ai sorgenti:
#  il progetto sta in "/mnt/c/.../Progetto Medical", e quello SPAZIO nel
#  percorso rompe il caricatore di mesh di RViz (basato su libcurl, che rifiuta
#  gli spazi non codificati in un URL file://). I marker MESH_RESOURCE
#  fallivano tutti con "Error retrieving file". Installando sotto la home di
#  Linux il percorso non ha spazi e le mesh si caricano; in piu' il filesystem
#  nativo e' molto piu' veloce di /mnt/c.
# ============================================================================
set -e
source /opt/ros/humble/setup.bash
WS="${CK_WS:-$HOME/ck_ws}"
cd "$(dirname "$(readlink -f "$0")")"
colcon build --symlink-install \
  --build-base "$WS/build" --install-base "$WS/install" "$@"
echo
echo "Compilato in $WS. Per usarlo:"
echo "  source $WS/install/setup.bash"
echo "  ros2 launch cyberknife_ros cyberknife.launch.py"

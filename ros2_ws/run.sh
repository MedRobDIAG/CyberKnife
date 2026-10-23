#!/usr/bin/env bash
# ============================================================================
#  Compila (se serve) e avvia la simulazione ROS 2. Va lanciato DA WSL.
#  Lo usa anche il pulsante "Lancia RViz" del Tab 3.
#      bash ros2_ws/run.sh [argomenti del launch]
# ============================================================================
set -e
source /opt/ros/humble/setup.bash
HERE="$(dirname "$(readlink -f "$0")")"
WS="${CK_WS:-$HOME/ck_ws}"
[ -f "$WS/install/setup.bash" ] || bash "$HERE/build.sh"
source "$WS/install/setup.bash"
# Un piano o una scena rigenerati su Windows non entrano nell'installazione
# finche' non si ricompila: con --symlink-install basta, ma se il pacchetto e'
# stato installato prima che i file esistessero i link mancano.
[ -f "$WS/install/cyberknife_ros/share/cyberknife_ros/plans/piano_trattamento.json" ] \
  || bash "$HERE/build.sh"
exec ros2 launch cyberknife_ros cyberknife.launch.py "$@"

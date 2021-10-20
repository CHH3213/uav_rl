#!/bin/bash
# 运行后会打开仿真，加载mavros
gnome-terminal --tab --title="killall gzclient" -- bash -c "killall gzclient;sleep 1; "
gnome-terminal --tab --title="killall gzserver" -- bash -c "killall gzserver;sleep 1; "
gnome-terminal --tab --title="SITL-gazebo" -- bash -c "roslaunch uav_rl runway.launch;exec bash;"
gnome-terminal --tab -- bash -c "cd ~/ardupilot/ArduCopter/ && sim_vehicle.py -v ArduCopter -f gazebo-iris;sleep 1; "
gnome-terminal --tab --title="Mavros" -- bash -c "roslaunch uav_rl apm.launch;exec bash;"



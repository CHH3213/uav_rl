#!/bin/bash
cd /home/chh3213/ardupilot/Tools/autotest/ && sim_vehicle.py -v ArduCopter -f gazebo-iris --console  -N

# -N 表示直接运行上次编译出的仿真固件

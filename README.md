(WIP: Readme.md)

Vision-based (RGB input only) Target Tracking and Following Using a UAV.


### RUNNING:

To run the main.py file, which requires:

1. Launching the default.sdf world file with walking actors for simulation environment with PX4-ROS2.
2. Offboard using <a href="https://github.com/PX4/px4_ros_com/blob/main/src/examples/offboard/offboard_control.cpp">px4_ros_com offboard example</a> <br>
3. Ultralytics ROS2 node which provides detection and tracking results in ROS2
<a href="https://github.com/Alpaca-zip/ultralytics_ros">Ultralytics ROS </a> <br>
Note: Use the humble-devel branch.
4. Finally, run the main.py file


### TODOs: <br>
TF Transforms (in ROS2-PX4) for UAV onboarded camera and map to get target's simulated world coordinates in Gazebo.

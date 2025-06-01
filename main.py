#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy, Duration
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from detection_msgs.msg import PublishData
from filterpy.kalman import KalmanFilter
import tf2_ros
import tf2_geometry_msgs

import numpy as np
import math
from cv_bridge import CvBridge
import time
import tf2_py as tf
import pyrealsense2 as rs2

from px4_msgs.msg import VehicleLocalPosition, SensorGps, VehicleAttitude, TrajectorySetpoint, VehicleGlobalPosition
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from ultralytics_ros.msg import YoloResult
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from scipy.spatial.transform import Rotation as R

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class BoundingBox2D:
    def __init__(self, center_x=0, center_y=0, size_x=0, size_y=0):
        self.center = Point(center_x, center_y)
        self.size_x = size_x
        self.size_y = size_y

class Detected_Object:
    def __init__(self, id, center_x, center_y, size_x, size_y):
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.size_x = size_x
        self.size_y = size_y
        
    def update(self, id, center_x, center_y, size_x, size_y):
        self.id = id
        self.center_x = center_x
        self.center_y = center_y
        self.size_x = size_x
        self.size_y = size_y

class LostCounter:
    def __init__(self, counting=False):
        self.counter = 30
        self.counting = counting
        self.lost = False
        print("Lost Counter Initialized")

    def update(self):
        if self.counter > 0:
            self.counter -= 1
            return True
        elif self.counter == 0:
            self.counting = False
            self.lost = True
            return False
            
    def check_counting(self):
        return self.counting
        
    def is_lost(self):
        return self.lost

class FollowTarget(Node):
    def __init__(self):
        super().__init__('follow_target')
        print("Node started")
        
        # Kalman Filter initialization
        self.KF = KalmanFilter(dim_x=6, dim_z=4)
        self.KF.H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
        self.KF.P *= 5
        self.KF.R = np.array([[2, 0, 0, 0],
                             [0, 2, 0, 0],
                             [0, 0, 20, 0],
                             [0, 0, 0, 20]])
        self.KF.Q = np.array([[2, 0, 0, 0, 0, 0],
                             [0, 2, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])

        self.KF_target = KalmanFilter(dim_x=4, dim_z=2)
        self.KF_target.H = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0]])
        self.KF_target.P *= 5
        self.KF_target.R = np.array([[100, 0],
                                   [0, 100]])
        self.KF_target.Q = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        # Initialize variables
        self.intrinsics = None
        self.target_position = Point()
        self.target_hdg = 0
        self.altitude = 0
        self.target_detected = False
        self.first_detection = False
        self.bridge = CvBridge()
        self.target_lost = False
        self.target_id = 0
        self.adjustment_period = 60
        self.lost_counter = LostCounter()
        
        # Using TrajectorySetpoint for PX4
        self.trajectory_setpoint = TrajectorySetpoint()
        
        # Controller parameters
        self.previous_error = 0.0
        self.integral = 0
        self.heading = 0
        self.target_hdg = 0
        
        # Controller gains
        self.kh = 0.5
        self.kh2 = 0.6
        self.kp_vel = 0.8
        self.kd_vel = 0.0
        self.ki_vel = 0.01
        self.kp_altitude = 1.2
        self.alpha = 0.10
        self.filtered_value = None
        
        # Slew rate limiter
        self.max_rate = 0.4
        self.last_update_time = time.time()
        self.last_update_time_KF = time.time()
        self.current_value = 0.0
        
        # Camera parameters
        self.fx = 0.1
        self.fy = 0.1
        self.ppx = 0
        self.ppy = 0
        self.camera_width = 1280
        self.camera_height = 720
        self.distance = 0
        self.dist_horizontal = 0
        self.pitch_camera = np.pi/6
        
        # UAV state
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.uav_pose = Point()
        
        # Data storage
        self.recent_distances = []
        self.detected_objects = []
        self.detected_objects_ids = []
        self.bool = True
        self.publish_data = PublishData()
        
  
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
         
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        print("--- QOS changed  checking---")

        self.odometry_sub = self.create_subscription(
            VehicleLocalPosition, 
            '/fmu/out/vehicle_local_position', 
            self.odometry_callback, 
            qos_profile=qos_profile)
            
    
        self.altitude_sub = self.create_subscription(
            VehicleGlobalPosition, 
            '/fmu/out/vehicle_global_position', 
            self.get_altitude, 
            qos_profile=qos_profile)
        

        self.target_sub = self.create_subscription(
            Detection2DArray,
            # YoloResult,
            '/detection_result', 
            self.target_callback, 
            # self.check_target,
            # qos_profile=qos_profile
            10)
        
            
        self.heading_sub = self.create_subscription(
            VehicleAttitude, 
            '/fmu/out/vehicle_attitude', 
            self.get_heading, 
            qos_profile=qos_profile)
        

        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera_info', 
            self.camera_info_callback, 
            10)
 
        # Publishers
        self.publisher = self.create_publisher(PublishData, '/rosbag_data', 10)
        self.publish_pose = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.publish_pose2 = self.create_publisher(PoseStamped, '/target_pose2', 10)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, 
            '/fmu/in/trajectory_setpoint', 
            10)

    def odometry_callback(self, msg):
        self.uav_pose.x = msg.x
        self.uav_pose.y = msg.y
        self.uav_pose.z = msg.z
        # print(self.uav_pose.x)


    # TODO 
    def target_coordinates_test(self, distance, center_x, center_y, heading):
        """
        Calculate target coordinates in world frame based on camera detection.
        
        Args:
            distance: Estimated distance to target (meters)
            center_x: X coordinate of target center in image (pixels)
            center_y: Y coordinate of target center in image (pixels)
            heading: Current drone heading (radians)
            
        Returns:
            tuple: (target_x, target_y) in world coordinates
        """
        pass

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        half_w1, half_h1 = w1 / 2, h1 / 2
        half_w2, half_h2 = w2 / 2, h2 / 2

        x1_tl, y1_tl = x1 - half_w1, y1 - half_h1
        x1_br, y1_br = x1 + half_w1, y1 + half_h1
        x2_tl, y2_tl = x2 - half_w2, y2 - half_h2
        x2_br, y2_br = x2 + half_w2, y2 + half_h2

        x_inter = max(x1_tl, x2_tl)
        y_inter = max(y1_tl, y2_tl)
        x2_inter = min(x1_br, x2_br)
        y2_inter = min(y1_br, y2_br)

        intersection_area = max(0, x2_inter - x_inter) * max(0, y2_inter - y_inter)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def get_altitude(self, msg):
        self.altitude = msg.alt
        # print(self.altitude)

    def create_velocity_msg(self, Vx, Vy, Vz, yaw_rate):
        msg = TrajectorySetpoint()
        msg.velocity[0] = Vx
        msg.velocity[2] = -Vz
        msg.yawspeed = float(yaw_rate)        
        self.target_detected = True

        print(f"Setpoint NED: Vx={Vx}, Vy={Vy}, Vz={Vz}")
        return msg
    

    def RateLimiter(self, new_value):
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        self.last_update_time = current_time

        if time_elapsed == 0:
            return self.current_value

        rate_of_change = (new_value - self.current_value) / time_elapsed

        if self.bool:
            self.current_value = 0
            self.bool = False
            return self.current_value

        if abs(rate_of_change) <= self.max_rate:
            self.current_value = new_value
        else:
            sign = 1 if new_value > self.current_value else -1
            self.current_value += sign * self.max_rate * time_elapsed
            
        return self.current_value

    def low_pass_filter(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

    def camera_info_callback(self, msg):
        try:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.ppx = msg.k[2]
            self.ppy = msg.k[5]

            # Initialize realsense intrinsics
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = msg.width
            self.intrinsics.height = msg.height
            self.intrinsics.ppx = self.ppx
            self.intrinsics.ppy = self.ppy
            self.intrinsics.fx = self.fx
            self.intrinsics.fy = self.fy
            
            if msg.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif msg.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
                
            self.intrinsics.coeffs = [i for i in msg.d]
            
            # self.get_logger().info("Camera intrinsics initialized")
        except Exception as e:
            self.get_logger().error(f'Camera info error: {str(e)}')
            self.intrinsics = None

    def get_heading(self, msg):
        # Convert quaternion to Euler angles
        q = msg.q
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.heading = math.atan2(siny_cosp, cosy_cosp)
       
    def velocity_controller(self, error):
        derivative = error - self.previous_error
        self.integral += error 
        self.integral = np.clip(self.integral, -2, 2)
        self.previous_error = error
        
        control_output = (error * self.kp_vel + 
                         self.kd_vel * derivative + 
                         self.ki_vel * self.integral)
        control_output = np.clip(control_output, -4, 4)
        control_output = self.RateLimiter(control_output)

        if self.dist_horizontal < 8 and control_output > 0:
            control_output = control_output/2
        elif self.dist_horizontal < 6 and control_output > 0:
            control_output = 0

        print("horizontal distance", self.dist_horizontal)

        self.publish_data.control_output = float(control_output)

        normalized_y = ((self.KF.x[1] / self.ppy) - 1)
        
        Vx = math.cos(self.heading) * control_output
        Vy = math.sin(self.heading) * control_output
        Vz = -self.kh2 * normalized_y
        
        # Altitude limits
        if self.altitude > 9 and Vz > 0:
            Vz = 0
        elif self.altitude < 3 and Vz < 0:
            Vz = 0
        
        trajectory_msg = self.create_trajectory_setpoint(
            Vx, Vy, Vz, 
            self.target_hdg, 
            self.target_altitude)

        return trajectory_msg

    def Update_Objects(self, data, target_found):
        target_bbox = BoundingBox2D()  
        best_match_value = 0
        best_detection = None
        threshold = 0.1
    
        if not(target_found):
            my_target = next((obj for obj in self.detected_objects if obj.id == self.target_id), None)
                
            my_target_position_increased = (my_target.center_x, my_target.center_y, my_target.size_x + 250, my_target.size_y + 150)

            target_bbox.center.x = self.KF.x[0]
            target_bbox.center.y = self.KF.x[1]
            target_bbox.size_x = self.KF.x[4]
            target_bbox.size_y = self.KF.x[5]
            threshold = 2 * target_bbox.size_y

            if not(self.target_lost) and not(self.lost_counter.check_counting()):
                self.target_lost = True
                self.lost_counter.counting = True
                self.lost_counter.counter = 30
                print("START LOST COUNTER, CHECK =", self.lost_counter.check_counting())
            else:
                self.lost_counter.update()
        else:
            self.lost_counter.counting = False
            self.lost_counter.counter = 30
            self.lost_counter.lost = False
            print("target found, RESET COUNTER, CHECK =", self.lost_counter.check_counting())

        for detection in data.detections:
            bounding_box = detection.bbox
            # id = detection.results[0].id
            id = detection.id
            center_x = bounding_box.center.position.x
            center_y = bounding_box.center.position.y
            # center_x = bounding_box.center.x
            # center_y = bounding_box.center.y

            size_x = bounding_box.size_x
            size_y = bounding_box.size_y

            existing_object = next((obj for obj in self.detected_objects if obj.id == id), None)

            if existing_object:
                existing_object.update(id, center_x, center_y, size_x, size_y)
                if existing_object.id == self.target_id:    
                    target_bbox = detection.bbox
                    self.target_lost = False
            else:
                if target_found:
                    if id != 0:
                        new_object = Detected_Object(id, center_x, center_y, size_x, size_y)
                        self.detected_objects.append(new_object)
                else:
                    if id != 0: 
                        iou = self.calculate_iou((center_x, center_y, size_x + 250, size_y + 150), my_target_position_increased)
                        print("Interception over union:", iou)
                        new_object = Detected_Object(id, center_x, center_y, size_x, size_y)
                        self.detected_objects.append(new_object)
                        if iou > best_match_value:
                            best_match_value = iou
                            best_detection = detection
                            best_id = id
                            print("best detection", id)

        if not(target_found) and best_detection is not None:
            if best_match_value > 0.1:
                my_target.update(best_id, center_x, center_y, size_x, size_y)
                delete = next((obj for obj in self.detected_objects if obj.id == self.target_id), None)
                if delete is not None:
                    delete.update(999, 0, 0, 0, 0)
                self.target_id = best_id
                target_bbox = best_detection.bbox
                target_found = 1
                self.target_lost = False
                if self.lost_counter.is_lost():
                    self.filtered_value = None
                    self.bool = True
                    self.lost_counter.lost = False
                    self.adjustment_period = 10
                    print("RESET VALUES")
                self.lost_counter.counting = False
                print("TARGET REDETECTED, RESET COUNTER", self.lost_counter.check_counting(), best_id)
                self.lost_counter.counter = 30
                for ii in self.detected_objects:
                    print("\n", ii.id)
            else:
                target_found = 0
                print("No suitable match found.", threshold, best_match_value)                
    
        return target_bbox
    
    def target_callback(self, data):
            print("-----------------------")
            # Process the target detection data and update self.target_position and self.target_detected
            # Extract the bounding box information from the message

            #___________________________________Kalman_Filter_______________________________________	
            #Constant Velocity Model	
            current_time = time.time()
            dt = current_time - self.last_update_time_KF
            self.last_update_time_KF = current_time
            self.KF.F = np.array([[1, 0, dt, 0, 0, 0],
                                [0, 1, 0, dt, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
            self.KF.predict()

            self.KF_target.F = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
            
            self.KF_target.predict()

            if not(data.detections or self.target_lost):
                self.KF.update(None)
                self.KF_target.update(None)
            #________________________________________________________________________________________
                

            if not(data.detections) and self.first_detection:
                self.first_detection = True
                self.target_lost = True

            if data.detections or self.target_lost:
                # Loop through all detected objects
                
                

                #Initilize target on first detection
                if not(self.first_detection):
                    self.target_id = data.detections[0].id
                    print("target detected id first frame:\n", self.target_id)
                    self.first_detection = True

 
                    center = data.detections[0].bbox.center.position
                    size = data.detections[0].bbox
                    new_object = Detected_Object(
                        self.target_id,
                        center.x,  # Use position.x instead of center.x
                        center.y,  # Use position.y instead of center.y
                        size.size_x,    # Use size_x or size.x depending on your message definition
                        size.size_y     # Use size_y or size.y depending on your message definition
                    )
                    self.detected_objects.append(new_object)

                # Check if the target is in the detection message
                target_found = 0
                for detection in data.detections:
                    if detection.id == self.target_id:
                        target_found = 1
                if target_found==0:
                    target_found=False
                elif target_found==1:
                    target_found=True
                self.publish_data.target_lost = target_found
                        
                #Update the position of all objects and get target bbox information
                #If target was not found in detection, do Id Switch Recovery
                bounding_box = self.Update_Objects(data, target_found)	

                if bounding_box is not None:

                    if hasattr(bounding_box.center, 'position'):
                        # Pose2D case
                        center_x = bounding_box.center.position.x
                        center_y = bounding_box.center.position.y
                        # print(center_x)
                    else:
                        # Point case
                        center_x = bounding_box.center.x
                        center_y = bounding_box.center.y
                        
                    size_x = bounding_box.size_x
                    size_y = bounding_box.size_y

                    if not(self.target_lost):
                        self.KF.update([center_x, center_y, size_x, size_y])
                    self.publish_data.distance_cnst = float(self.KF.x[0])
                    self.publish_data.distance_singer = float(self.KF.x[1])


                    # Convert center coordinates to normalized coordinates (-1 to 1)
                    normalized_x = (self.KF.x[0] / self.ppx) - 1
                    normalized_y = (self.KF.x[1] / self.ppy) - 1

                    size_y = self.KF.x[5]

                    #Calculate target heading - put bbox in the center of the frame
                    self.target_hdg = - self.kh * (normalized_x)
                    
                    self.publish_data.error_heading = - float(self.kh * (normalized_x))
                    self.publish_data.id = int(self.target_id)

                    # Calculate the error between the target and the center of the image
                    self.target_altitude = self.altitude - self.kp_altitude * (normalized_y)# - (self.altitude - 6) * 0.1

                    self.publish_data.error_altitude = - float(self.kp_altitude * (normalized_y))# - (self.altitude - 6) * 0.1

                    if self.target_altitude < 3:
                        self.target_altitude = 3
                    elif self.target_altitude > 10:
                        self.target_altitude = 10

                    if size_y != 0:
                        distance = 1450/(size_y)
                    else:
                        distance = 10
                    
                    value = self.low_pass_filter(distance)

                    #target_x, target_y = self.target_world_coordinates(value, self.target_hdg)
                    #if self.not_using_depth:
                    target_x, target_y = self.target_coordinates_test(value,center_x,center_y,self.heading)
                    self.KF_target.update([target_x, target_y])

                    self.publish_data.target_x = float(target_x)
                    self.publish_data.target_y = float(target_y)
                    self.publish_data.estimated_target_x = float(self.KF_target.x[0])
                    self.publish_data.estimated_target_y = float(self.KF_target.x[1])

                    estimated_distance = value
                    #print("estimated distance", estimated_distance)
                    error = estimated_distance - 11 #2*self.altitude + 1 #11 #

                    self.dist_horizontal = np.sqrt(value**2 - (self.altitude)**2)
                    



                    #self.publish_data.altitude = self.altitude
                    self.publish_data.estimated_distance= float(estimated_distance)
                    self.publish_data.use_depth = False
                    self.publish_data.center_x = float(center_x)
                    self.publish_data.center_y = float(center_y)
                    self.publish_data.controller_error = float(error)

                    if (self.adjustment_period > 0):
                        if not(self.target_lost):
                            print("initial")
                            self.adjustment_period -= 1	
                        print("ADJUSTING", self.target_id)				
                        # self.velocity_msg = self.create_velocity_msg(0, 0, 0, self.target_hdg, self.target_altitude, True)
                        #print("adjusting", self.target_hdg, self.target_altitude)	
                        self.trajectory_setpoint = self.velocity_msg(0, 0, 0, self.target_hdg, self.target_altitude)
                
                    elif self.target_lost and not(self.lost_counter.check_counting()) and self.lost_counter.is_lost():
                        print("---- LOST ----", self.target_id , self.lost_counter.counter)
                        
                        control_output = self.RateLimiter(0)
                        self.publish_data.control_output = float(control_output)	
                        Vx = math.cos(self.heading) * control_output
                        Vy = math.sin(self.heading) * control_output
                        # self.velocity_msg = self.create_velocity_msg(Vx, Vy, 0, 0, self.lost_altitude, True)
                        self.trajectory_setpoint = self.velocity_msg(Vx, Vy, 0, 0, self.lost_altitude)
    
                    else:
                        if self.target_lost:
                            self.velocity_msg = self.velocity_controller(error - 1)
                            self.lost_altitude = self.altitude
                            print("---- MISSING ----",  self.target_id , self.lost_counter.counter, error)
                            # print("designated heading part2 =",self.velocity_msg.heading)
                        else:
                            self.velocity_msg = self.velocity_controller(error)
                            print("---- REGULAR ----", self.target_id)

            else: #not(self.first_detection):
                self.trajectory_setpoint = self.velocity_msg(0, 0, 0, 0.1, 4)
                

def main(args=None):
    rclpy.init(args=args)
    follower = FollowTarget()
    
    try:
        print("Starting spin...")  # Debug
        rclpy.spin(follower)  
    except KeyboardInterrupt:
        pass
    finally:
        follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
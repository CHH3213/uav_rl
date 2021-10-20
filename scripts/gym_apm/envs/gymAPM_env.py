#!/usr/bin/env python3

import math
import numpy as np
import rospy
import time
import sys
import socket
import pickle

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Vector3

from mavros_msgs.msg import ActuatorControl
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import Thrust
from mavros_msgs.msg import State
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import SetModeRequest
from mavros_msgs.srv import SetModeResponse
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandTOL
from mavros_msgs.srv import CommandBoolRequest
from mavros_msgs.srv import CommandBoolResponse
from mavros_msgs.srv import StreamRate, StreamRateRequest

from sensor_msgs.msg import Imu
from sensor_msgs.msg import BatteryState

from std_msgs.msg import Header
from std_msgs.msg import Float64
from std_srvs.srv import Empty, EmptyRequest

import mavros.setpoint

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from gazebo_msgs.msg import ModelState, ModelStates

import gym
from gym import spaces
import math


def counter_cosine_similarity(c1, c2):
    '''
    计算余弦相似度
    '''
    from collections import Counter
    c1 = Counter(c1)
    c2 = Counter(c2)
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    return dotprod / (magA * magB)


def quaternion_to_euler(x, y, z, w):
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # X = math.degrees(math.atan2(t0, t1))
    #
    # t2 = +2.0 * (w * y - z * x)
    # # t2 = +1.0 if t2 > +1.0 else t2
    # # t2 = -1.0 if t2 < -1.0 else t2
    # Y = math.degrees(math.asin(t2))
    #
    # t3 = +2.0 * (w * z + x * y)
    # t4 = +1.0 - 2.0 * (y * y + z * z)
    # Z = math.degrees(math.atan2(t3, t4))

    # 使用 tf 库
    import tf
    (X, Y, Z) = tf.transformations.euler_from_quaternion([x, y, z, w])
    return X, Y, Z


def euler_to_quaternion(roll, pitch, yaw):
    # x=sin(pitch/2)sin(yaw/2)cos(roll/2)+cos(pitch/2)cos(yaw/2)sin(roll/2)
    # y=sin(pitch/2)cos(yaw/2)cos(roll/2)+cos(pitch/2)sin(yaw/2)sin(roll/2)
    # z=cos(pitch/2)sin(yaw/2)cos(roll/2)-sin(pitch/2)cos(yaw/2)sin(roll/2)
    # w=cos(pitch/2)cos(yaw/2)cos(roll/2)-sin(pitch/2)sin(yaw/2)sin(roll/2)

    import tf
    (x, y, z, w) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

    return x, y, z, w


class gymAPM(gym.Env):

    def __init__(self):

        ### r,p,y,thrust
        # 或者使用RC控制每一个螺旋桨电机，范围为【1000，2000】
        self.min_action = np.array([-1, -1, -1, 0])
        self.max_action = np.array([1, 1, 1, 1])
        self.act_dim = len(self.min_action)
        # print(self.act_dim)
        # 高度
        self.min_altitude = 0.1
        self.max_altitude = 25
        # 角度
        self.min_roll = -180
        self.min_pitch = -180
        self.min_yaw = 0
        self.max_roll = 180
        self.max_pitch = 180
        self.max_yaw = 360
        # 角速度
        self.min_rollRate = -60
        self.min_pitchRate = -60
        self.min_yawRate = -60
        self.max_rollRate = -self.min_rollRate
        self.max_pitchRate = -self.min_pitchRate
        self.max_yawRate = -self.min_yawRate
        # 线速度
        self.min_rollSpeed = -3
        self.min_pitchSpeed = -3
        self.min_yawSpeed = -3
        self.max_rollSpeed = -self.min_rollSpeed
        self.max_pitchSpeed = -self.min_pitchSpeed
        self.max_yawSpeed = -self.min_yawSpeed

        self.low_state = np.array(
            [self.min_altitude, self.min_roll, self.min_pitch, self.min_yaw, self.min_rollRate, self.min_pitchRate,
             self.min_yawRate, self.min_rollSpeed, self.min_pitchSpeed, self.min_yawSpeed])

        self.high_state = np.array(
            [self.max_altitude, self.max_roll, self.max_pitch, self.max_yaw, self.max_rollRate, self.max_pitchRate,
             self.max_yawRate, self.max_rollSpeed, self.max_pitchSpeed, self.max_yawSpeed])

        self.obs_dim = len(self.low_state)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)
        # observation: 高度1，姿态（欧拉角）3，角速度3，线速度3
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        self.desired_state = np.array([0, 0, 0, 0, 0, 0])  # 姿态平稳：角度和角速度都为0

        ###  define ROS messages  ######
        self.current_state = State()
        self.imu_data = Imu()
        self.act_controls = ActuatorControl()
        self.pose = PoseStamped()
        self.mocap_pose = PoseStamped()
        self.thrust_ctrl = Thrust()
        self.attitude_target = AttitudeTarget()
        self.local_velocity = TwistStamped()
        self.global_velocity = TwistStamped()
        self.battery = BatteryState()
        self.local_position = PoseStamped()
        self.model_states = ModelStates()
        ### ############################## ###

        ### Initiate ROS node
        print('-- Connecting to mavros')
        rospy.init_node('gym_apm_mavros', anonymous=True)
        print('connected')

        ## ROS Subscribers
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb, queue_size=1)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_cb, queue_size=1)
        self.local_pos_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.lp_cb, queue_size=1)
        self.local_vel_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.lv_cb,
                                              queue_size=1)
        self.act_control_sub = rospy.Subscriber("/mavros/act_control/act_control_pub", ActuatorControl, self.act_cb,
                                                queue_size=1)
        self.global_alt_sub = rospy.Subscriber("/mavros/global_position/rel_alt", Float64, self.ra_cb, queue_size=1)
        self.global_pos_sub = rospy.Subscriber("/mavros/global_position/gp_vel", TwistStamped, self.gv_cb, queue_size=1)

        self.battery_sub = rospy.Subscriber("/mavros/battery", BatteryState, self.bat_cb, queue_size=1)

        ## ROS Publishers
        # self.mocap_pos_pub = rospy.Publisher("/mavros/mocap/pose",PoseStamped,queue_size=1)
        self.acutator_control_pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl,
                                                    queue_size=1)  # 使用r,p,y,thrust控制
        self.setpoint_raw_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)
        self.setcmd_vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=1)
        self.thrust_pub = rospy.Publisher("/mavros/setpoint_attitude/thrust", Thrust, queue_size=1)
        self.setattitude_pub = rospy.Publisher("/mavros/setpoint_attitude/attitude", PoseStamped, queue_size=1)
        self.setRcControl_pub = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=1)  # 使用rc控制
        self.rc_override = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=1)

        ## ROS mavros Services
        rospy.wait_for_service('mavros/cmd/arming')
        self.arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)

        rospy.wait_for_service('mavros/cmd/takeoff')
        self.takeoff_service = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)

        rospy.wait_for_service('mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

        '''########################'''
        rospy.wait_for_service('mavros/set_stream_rate')
        set_stream_rate = rospy.ServiceProxy("mavros/set_stream_rate", StreamRate)
        set_stream_rate(StreamRateRequest.STREAM_POSITION, 50, 1)
        set_stream_rate(StreamRateRequest.STREAM_ALL, 50, 1)
        #########################
        # 姿态信息的发布重置环境时使用
        self.setpoint_msg = mavros.setpoint.PoseStamped(
            header=mavros.setpoint.Header(frame_id="att_pose", stamp=rospy.Time.now()), )

        # ############################## gazebo services##########################
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # 指定服务名来调用服务
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        ###################################################################
        self.modestate_sub=rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)

    def get_state(self, model_state, model_name):
        model = model_state.name.index(model_name)  #

        model_pose = model_state.pose[model]
        model_twist = model_state.twist[model]
        model_position = [model_pose.position.x, model_pose.position.y, model_pose.position.z]
        roll, pitch, yaw = quaternion_to_euler(model_pose.orientation.x, model_pose.orientation.y, model_pose.orientation.z, model_pose.orientation.w)
        model_attitude = [roll, pitch, yaw]
        model_linear = [model_twist.linear.x, model_twist.linear.y, model_twist.linear.z]
        model_angular = [model_twist.angular.x, model_twist.angular.y, model_twist.angular.z]
        # print([model_position,model_orientation,model_linear,model_angular])
        # 位置，姿态，线速度，角速度
        return [model_position, model_attitude, model_angular, model_linear ]
        # 位置与姿态
        # return [model_position,model_attitude]
    # 使用gazebo服务重置
    def resetWorld(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def resetSim(self):
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_simulation()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

    def pausePhysics(self):
        rospy.wait_for_service("/gazebo/pause_physics")  # 等待服务器连接
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

    def unpausePhysics(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

    def reset(self):
        '''
        针对RC控制的情况
        Reset environment
        Usage:
            obs = env.reset()

        '''
        self.unpausePhysics()
        self.success_steps = 0
        self.steps = 0
        # initial state
        self.resetWorld()
        # self.reset_sim()
        # self.send_msg_get_return('reset')  # 测试
        print('sleep 3s')
        time.sleep(0.1)
        print('-- Resetting position')
        '''mavros获取'''
        # 四元数
        qx = self.local_position.pose.orientation.x
        qy = self.local_position.pose.orientation.y
        qz = self.local_position.pose.orientation.z
        qw = self.local_position.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        # 位置
        x = self.local_position.pose.position.x
        y = self.local_position.pose.position.y
        z = self.local_position.pose.position.z
        # 线速度
        vx = self.local_velocity.twist.linear.x
        vy = self.local_velocity.twist.linear.y
        vz = self.local_velocity.twist.linear.z
        # 角速度
        ang_vx = self.local_velocity.twist.angular.x
        ang_vy = self.local_velocity.twist.angular.y
        ang_vz = self.local_velocity.twist.angular.z
        self.state = np.array([z, roll, pitch, yaw, ang_vx, ang_vy, ang_vz, vx, vy, vz])
        '''
        gazebo获取
        '''
        iris_state = self .get_state(self.model_states, 'drone_with_camera')
        iris_state = np.array(iris_state).flatten()  # 位置，姿态(euler)，角速度,线速度
        self.state = iris_state[2:]  # 只需要高度，不需要平面位置

        return self.state

    # def reset(self):
    #     '''
    #     针对不是RC控制的情况
    #     Reset environment
    #     Usage:
    #         obs = env.reset()
    #     '''
    #     # 解锁---先飞到某个高度，再开始训练
    #     self.offb_arm()
    #     while True:
    #         if self.current_state.armed == False:
    #             self.offb_arm()
    #             self.takeoff_service(altitude=10)
    #         if np.abs(self.local_position.pose.position.z - 20) < 0.5:
    #             break
    #
    #     self.success_steps = 0
    #     self.steps=0
    #     # reset_steps = 0
    #     # initial state
    #     self.initial = [20,np.random.uniform(-50,50),np.random.uniform(-50,50),np.random.uniform(-50,50),np.random.uniform(-50,50),
    #                     np.random.uniform(-50,50),np.random.uniform(-50,50),np.random.uniform(-3,3),np.random.uniform(-3,3),np.random.uniform(-3,3)]
    #
    #     print('Initial: ', self.initial[1:7],'\n', 'Desired: ', self.desired_state)
    #
    #     print('-- Resetting position')
    #
    #     if self.current_state.armed == False:
    #         self.offb_arm()
    #
    #     # r = rospy.Rate(50)
    #     #姿态
    #     attitude_pub = PoseStamped()
    #     x,y,z,w = euler_to_quaternion(self.initial[1],self.initial[2],self.initial[3])
    #     self.setpoint_msg.pose.orientation.x = x
    #     self.setpoint_msg.pose.orientation.y = y
    #     self.setpoint_msg.pose.orientation.z = z
    #     self.setpoint_msg.pose.orientation.w = w
    #     self.local_pos_pub.publish(self.setpoint_msg)
    #     # 速度
    #     vel_pub = Twist()
    #     vel_pub.angular.x = self.initial[4]
    #     vel_pub.angular.y = self.initial[5]
    #     vel_pub.angular.z = self.initial[6]
    #     vel_pub.linear.x = self.initial[7]
    #     vel_pub.linear.y = self.initial[8]
    #     vel_pub.linear.z = self.initial[9]
    #     self.setcmd_vel_pub.publish(vel_pub)
    #
    #     print('-------- Position reset --------')
    #
    #     self.state = np.array(self.initial)
    #
    #     return self.state

    def step(self, action):
        # print(action)
        self.unpausePhysics()
        start_time = time.time()
        # rate = rospy.Rate(50)
        '''MAVROS'''
        # 四元数
        qx = self.local_position.pose.orientation.x
        qy = self.local_position.pose.orientation.y
        qz = self.local_position.pose.orientation.z
        qw = self.local_position.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        ang_pos = [roll, pitch, yaw]
        # 位置
        x = self.local_position.pose.position.x
        y = self.local_position.pose.position.y
        z = self.local_position.pose.position.z
        lin_pos = [x, y, z]
        # 线速度
        vx = self.local_velocity.twist.linear.x
        vy = self.local_velocity.twist.linear.y
        vz = self.local_velocity.twist.linear.z
        lin_vel = [vx, vy, vz]
        # 角速度
        ang_vx = self.local_velocity.twist.angular.x
        ang_vy = self.local_velocity.twist.angular.y
        ang_vz = self.local_velocity.twist.angular.z
        ang_vel = [ang_vx, ang_vy, ang_vz]
        self.state = np.array([z, roll, pitch, yaw, ang_vx, ang_vy, ang_vz, vx, vy, vz])

        # '''法一：send actuator control commands'''
        # #actuator_control为无人机底层控制量（Mx My Mz 及 F） [0][1][2][3]分别对应 roll pitch yaw控制量 及 油门推力
        # self.act_controls.controls[0]=action[0]
        # self.act_controls.controls[1]=action[1]
        # self.act_controls.controls[2]=action[2]
        # # 训练得到的action均为[-1,1]，但是推力为[0,1]，故需要重映射
        # act3 = (action[3]-(-1))*(1-0)/(1-(-1))+0
        # self.act_controls.controls[3]=act3
        # self.acutator_control_pub.publish(self.act_controls)

        '''
        法二，action为每个螺旋桨的pwm,需要提前修改将前面4个映射成每个螺旋桨的pwm
        '''
        # 重映射成pwm--1000~2000，整数
        pwm1 = int((action[0] - (-1)) * (2000 - 1000) / (1 - (-1))) + 1000
        pwm2 = int((action[1] - (-1)) * (2000 - 1000) / (1 - (-1))) + 1000
        pwm3 = int((action[2] - (-1)) * (2000 - 1000) / (1 - (-1))) + 1000
        pwm4 = int((action[3] - (-1)) * (2000 - 1000) / (1 - (-1))) + 1000

        RC_PWM = OverrideRCIn()
        RC_PWM.channels = [pwm1, pwm2, pwm3, pwm4, 1500, 1500, 1500, 1500]
        self.setRcControl_pub.publish(RC_PWM.channels)

        reward, done = self.reward_done()
        # 频率问题
        step_prelen = time.time() - start_time
        if step_prelen < 0.03:
            time.sleep(0.03 - step_prelen)

        # 每一个step的时间
        step_len = time.time() - start_time

        # print('state: ', self.state , 'action: ', action , 'reward: ', reward, 'time: ', step_len)
        # print("battery", self.battery.percentage)
        '''
        gazebo获取
        '''
        iris_state = self.get_state(self.model_states, 'drone_with_camera')
        iris_state = np.array(iris_state).flatten()  # 位置，姿态(euler)，角速度,线速度
        self.state = iris_state[2:]  # 只需要高度，不需要平面位置
        info = {"state": self.state, "action": action, "reward": reward, "step": self.steps, "step length": step_len}
        return self.state, reward, done, info

    def reward_done(self):
        delt_dist = np.linalg.norm(np.array(self.desired_state) - np.array(self.state[1:7]))
        cos_similarity = counter_cosine_similarity(self.desired_state, self.state[1:7])
        reward = -0.1 * delt_dist - cos_similarity
        done = False

        if delt_dist < 0.5 and cos_similarity < 0.01:
            self.success_steps += 1
            if self.success_steps > 50:
                done = True
                reward += 1000

        return reward, done

    def land(self):
        """
        Set in LAND mode, which should cause the UAV to descend directly,
        land, and disarm.
        """
        self.set_mode_client(custom_mode="9")
        self.arming_client(False)

    def offb_arm(self):

        # print ('-- Enabling offboard mode and arming')
        # while not self.arm_cmd.value:
        #     pass
        # mode 0 = STABILIZE
        # mode 4 = GUIDED
        # mode 9 = LAND
        self.set_mode_client(custom_mode="4")
        self.arming_client(True)

        # rospy.loginfo('-- Ready to fly')

    def render(self):
        pass

    def close(self):
        pass

    def lv_cb(self, data):
        self.local_velocity = data

    def lp_cb(self, data):
        self.local_position = data

    def state_cb(self, data):
        self.current_state = data

    def imu_cb(self, data):
        self.imu_data = data

    def act_cb(self, data):
        self.act_controls = data

    def gv_cb(self, data):
        self.global_velocity = data

    def ra_cb(self, data):
        self.relative_altitude = data

    def bat_cb(self, data):
        self.battery = data
    def model_states_cb(self,data):
        self.model_states  = data
    def send_msg_get_return(self, msg):
        '''
        直接使用socket重置，测试中。。。
        '''
        ctrl_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                # print('@env@ try to connect with ctrl server')
                ctrl_client.connect(('localhost', 19881))
                connected = True
                # print('@env@ connected with ctrl server')
            except BaseException as e:
                # print('@env@[Error] ' + str(e))
                time.sleep(1)
                pass
        # done = False
        # while not done:
        # print('@env@ send msg: ' + msg)
        try:
            ctrl_client.send(msg)
            data = pickle.loads(ctrl_client.recv(1024))
            # print('@env@ send msg ' + msg + ' get return: ' + str(data))
            # done = True
        except BaseException as e:
            print('@env@[Error] ' + str(e))
            time.sleep(1)
        ctrl_client.close()
        return data

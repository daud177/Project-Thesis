import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from human_feedback import correct_action
from utils import KeyboardObserver, TrajectoriesDataset, loop_sleep
from custom_env import CustomEnv
from move_client import move_client
import threading
from RosRobot import RosRobot
from control_wsg_50 import control_wsg_50


   
    

def main(config):
    
    robot = RosRobot() # ANPASSUNG gmeiner
    robot.start_node() # ANPASSUNG gmeiner
    control_wsg50 = control_wsg_50() # ANPASSUNG gmeiner
    move_client_kuka = move_client() # ANPASSUNG gmeiner
    t_pos_start = threading.Thread(target=move_client_kuka.get_currentposition) # ANPASSUNG gmeiner
    t_pos_start.start() # ANPASSUNG gmeiner
    move_client_kuka.connect() # ANPASSUNG gmeiner
    move_client_kuka.move(-0.185,-0.65,0.355,180,0,90,"PTP",0,2,93) # self,x,y,z,roll ,pitch ,yaw ,movetyp, e1 , status ,turn 
    robot.start_launch('wsg_50_driver',"wsg_50_tcp_script.launch",'wsg_50')
    node_camera=robot.start_launch('halcon_matching',"realsense.launch",'camera','color_width:=320','color_height:=180','depth_width:=848','depth_height:=480','depth_fps:=30','color_fps:=30')
        
    
    save_path = "data/" + config["task"] + "/"
    assert os.path.exists(save_path)
    env = CustomEnv(config)
    keyboard_obs = KeyboardObserver()
    replay_memory = TrajectoriesDataset(config["sequence_len"])
    camera_obs, proprio_obs = env.reset()
    gripper_open = 0.9
    gripper_last_state = 0.9
    time.sleep(2)
    print("Go!")
    episodes_count = 0
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])#
    while episodes_count < config["episodes"]:
        start_time = time.time()
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9])#        
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            move_client_kuka.move_relativ(action)#
            gripper_open = action[-1]
            print("wann")
            time.sleep(0.2)
        action[-1] = keyboard_obs.get_gripper() #
        if action[-1] == -0.9: # ANPASSUNG gmeiner 07.02
            if action[-1] != gripper_last_state:
            	print('zu')
            	control_wsg50.grasp(65,200,10) # ANPASSUNG Gmeiner 07.02            
        if action[-1] == 0.9: # ANPASSUNG gmeiner 07.02
            if action[-1] != gripper_last_state:
            	control_wsg50.move(110,200) # ANPASSUNG Gmeiner 07.02  
            	print('auf')
        gripper_last_state = action[-1]          
        #print(action)
        next_camera_obs, next_proprio_obs, reward, done = env.step(action)
        replay_memory.add(camera_obs, proprio_obs, action, [1])
        camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
        if keyboard_obs.episode_reached_button: # ANPASSUNG gmeiner 07.02
            done = True # ANPASSUNG Gmeiner 07.02
            #env.gripper_plot.reset() #
            #gripper_open = 0.9
            #keyboard_obs.reset()
            #gpl.reset()
        if keyboard_obs.reset_button:
            replay_memory.reset_current_traj()
            camera_obs, proprio_obs = env.reset()
            gripper_open = 0.9
            keyboard_obs.reset()
        elif done:
            replay_memory.save_current_traj()
            camera_obs, proprio_obs = env.reset()
            gripper_open = 0.9
            if action[-1] == -0.9: # ANPASSUNG gmeiner 07.02
            	    control_wsg50.move(110,200) # ANPASSUNG Gmeiner 07.02  
            #control_wsg50.move(110,200) #alt Error-Vermeidung
            episodes_count += 1
            move_client_kuka.move(-0.185,-0.65,0.355,180,0,90,"LIN",0,2,93) # self,x,y,z,roll ,pitch ,yaw ,movetyp, e1 , status ,turn            
            print("Starten der neuen Episode mit Taste 'n', aktuelle Episode: ", episodes_count) # ANPASSUNG gmeiner 07.02
            keyboard_obs.wait_new_episode() # ANPASSUNG gmeiner 07.02
            keyboard_obs.reset()
            done = False
        else:
            loop_sleep(start_time)
    file_name = "demos_" + str(config["episodes"]) + ".dat"
    if config["save_demos"]:
        torch.save(replay_memory, save_path + file_name)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger, PickUpStatorReal",
    )
    args = parser.parse_args()
    config = {
        "task": args.task,
        "static_env": False,
        "headless_env": False,
        "save_demos": True,
        "episodes": 10,
        "sequence_len": 550,
    }
    main(config)

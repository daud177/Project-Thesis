# -*- coding: utf-8 -*-
"""
About the script:
An exmple on controlling KUKA iiwa robot from
Python3 using the iiwaPy3 class

Modified on 3rd-Jan-2021

@author: Mohammad SAFEEA

"""
import math
import time
import numpy as np
from datetime import datetime

from iiwaPy3 import iiwaPy3
from human_feedback import human_feedback
from utils import KeyboardObserver

start_time = datetime.now()


# returns the elapsed seconds since the start of the program
def getSecs():
    dt = datetime.now() - start_time
    secs = (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1000000.0
    return secs


ip = '172.31.1.147'
# ip='localhost'
iiwa = iiwaPy3(ip)
iiwa.setBlueOn()
time.sleep(2)
iiwa.setBlueOff()
key_board = KeyboardObserver()
# read some data from the robot
try:
    initVel = [0.1]
    #Move to an initial position    
    # initPos = [0, 0, 0, -math.pi / 2, 0, math.pi / 4, 0];
    # iiwa.movePTPJointSpace(initPos, initVel)

    # initPos2= [1, 0, 0, -math.pi / 2, 0, math.pi / 4, 0];
    # iiwa.movePTPJointSpace(initPos2, initVel)
    cPos = iiwa.getEEFPos()
    print("Current position", cPos)
    #key_board = KeyboardObserver()
    #print("keyboard", key_board.direction)
    while True: 
        #print("keyboard", key_board.direction)    
        newPos = key_board.direction
        print("New pos", newPos)
        if np.array_equal(newPos, np.array([0,0,0,0,0,0])) == False:
            newPos = np.append(newPos, 0)
            print(newPos)
            iiwa.movePTPJointSpace(newPos, initVel)
            print("movement Comlplete")
            #cPos = iiwa.getEEFPos()
        time.sleep(1)
        
        
except:
    print('an error happened')
# Close connection    
iiwa.close()

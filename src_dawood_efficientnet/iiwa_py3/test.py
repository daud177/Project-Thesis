import math
import time
from datetime import datetime

from iiwaPy3 import iiwaPy3
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
kobs = KeyboardObserver()


try:
    kk = kobs.get_direction()
    print("presses", kk)
    # Move to an initial position
    jPos = [0, 0, 0, math.pi / 4, 0, -math.pi / 4, 0]
    #print("Moving the robot in joint space to angular position")
    print(jPos)
    vRel = [0.1]
    print("With a relative velocity")
    print(vRel[0])
    #iiwa.movePTPJointSpace(jPos, vRel)
    
    
except:
    print('an error happened')

iiwa.close()

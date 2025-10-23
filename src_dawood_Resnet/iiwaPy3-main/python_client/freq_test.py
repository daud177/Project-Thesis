import math
import time
from datetime import datetime

from iiwaPy3 import iiwaPy3

start_time = datetime.now()


# returns the elapsed seconds since the start of the program
def getSecs():
    dt = datetime.now() - start_time
    secs = (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1000000.0
    return secs

def pr():
    print('inside the real time control of robot')
    do = 1
    return do


ip = '172.31.1.147'
# ip='localhost'
TPCtransform = (0, 0, 205.5, 0, 0, 0)  # (x,y,z,alfa,beta,gama)
iiwa = iiwaPy3(ip, TPCtransform)
# iiwa = iiwaPy3(ip)
iiwa.setBlueOn()
time.sleep(2)
iiwa.setBlueOff()
# robot control
try:

    # Move to an initial position    
    initPos = [0, 0, 0, -math.pi / 2, 0, math.pi / 2, 0];
    initVel = [0.1]
    iiwa.movePTPJointSpace(initPos, initVel)

    counter = 1
    index = 0  # index of x coordinate
    a = 0.9  # magnitude of motion [mm]

    print('here 42')
    pos = iiwa.getEEFPos()
    print('here 44')
    x0 = pos[index]
    print('initial TCP position')
    print(pos)
    iiwa.realTime_startDirectServoCartesian()
    #time.sleep(1)
    print('here 48')
    pos = iiwa.getEEFPos()
    print('here 50')
    
    t0 = getSecs()
    t_0 = getSecs()

    #wo = pr()
    while counter <= 100:
        pos[index] = x0 + a*counter

        if (getSecs() - t_0) > 0.04:
            jointPositions = iiwa.sendEEfPositionGetActualJpos(pos)
            t_0 = getSecs()
            counter = counter + 1
            #wo = pr()
            #print('inside the real time control of robot')

    deltat = getSecs() - t0;
    #time.sleep(1)

    iiwa.realTime_stopDirectServoCartesian()
    print('final TCP position')
    print(pos)
    print(f"counter = {counter-1}")
    print(f"total time = {deltat} s")
    print('update freq')
    print((counter-1) / deltat)

    # In our tests it achieved 260 Hz (for full duplex command-write-feedback-read) using:
    # Windows 10, intel i7 @3.6 GHz and Python 3.6.
    # This is without proper optimizations (also an Antivirus is on - couldn't disable it, university PC, requires admin pass to diable the Antivirus)
except:
    print('an error happened')
# Close connection
finally:    
    iiwa.close()

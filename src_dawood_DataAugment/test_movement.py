from move_client import move_client_iiwa_python
import math
import time
from datetime import datetime

start_time = datetime.now()

def getSecs():
    dt = datetime.now() - start_time
    secs = (dt.days * 24 * 60 * 60 + dt.seconds) + dt.microseconds / 1000000.0
    return secs

agent = move_client_iiwa_python()

iiwa = agent.connect()



try: 

    #agent.move()

    while True:
        agent.move([0, 0, 0, -math.pi / 2, 0, math.pi / 2, 0],[0.1])
        agent.get_currentposition()
        print('here line 26')

        agent.start_realtime()
        print('here line 29')
        agent.move_relativ([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
        print('here line 31')
        time.sleep(1)
        agent.stop_realtime()
        print('here line 33')
        a = input (" Press 'b' to stop or anthing else to proceed to Next Loop")

        if a == 'b':
            break


        



    # start_pos = agent.get_currentposition()

    # print('Starting position of end effector')
    # print(start_pos)


    # counter = 1
    
    # agent.start_realtime()
    
    # t_0 = getSecs()

    # # relative movement of 0.9 mm in x axis alone
    # relativ_pos = [1.0,0,0,0,0,0]

    
    # while counter <= 10:

    #     if (getSecs() - t_0) > 0.04:
    #         eff_pos = agent.move_relativ(relativ_pos)
    #         t_0 = getSecs()
    #         print(f'EEF position after command {counter} to move 0.9 mm in x-axis')
    #         print(eff_pos)
    #         counter = counter + 1

    # agent.stop_realtime()
    
        
except:
    print("Error")

finally:
    agent.disconnect()






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

    #agent.move([0, 0.285, 0, -1.5673, 0, 1.2666, 0],[0.1]) # stator pick position
    agent.move([0, 0.4475, 0, -1.76418, 0, 0.9070, 0],[0.1]) # socket pick position
    
        
except:
    print("Error")

finally:
    agent.disconnect()






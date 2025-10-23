import torch
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds



    
replay_memory_50_1 = torch.load("data/test_256_100/demos_50_1.dat")
replay_memory_50_2 = torch.load("data/test_256_100/demos_50_2.dat")

demos_100 = replay_memory_50_1

print(f'length of demos {len(demos_100)}')

for i in range(len(replay_memory_50_2)-2):
    camera_obs, proprio_obs, action, feedback=replay_memory_50_2[i]
    demos_100.camera_obs.append(camera_obs)
    demos_100.proprio_obs.append(proprio_obs)
    demos_100.action.append(action)
    demos_100.feedback.append(feedback)

print(f'length of demos {len(demos_100)}')
    

torch.save(demos_100, "demos_100.dat")
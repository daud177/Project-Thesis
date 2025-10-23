import torch
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds




# replay_memory_25 = torch.load("data/wire_bc/bc_25/demos_25.dat")
# replay_memory_50 = torch.load("data/wire_bc/bc_50/demos_25.dat")
# replay_memory_75 = torch.load("data/wire_bc/bc_75/demos_25.dat")
# replay_memory_100 = torch.load("data/wire_bc/bc_100/demos_25.dat")


# print(f'length of demos in 25_1 {len(replay_memory_25)}')

# for i in range(len(replay_memory_25)):
#     camera_obs, proprio_obs, action, feedback=replay_memory_25[i]
#     replay_memory_50.camera_obs.append(camera_obs)
#     replay_memory_50.proprio_obs.append(proprio_obs)
#     replay_memory_50.action.append(action)
#     replay_memory_50.feedback.append(feedback)

# print(f'length of demos in 50 {len(replay_memory_50)}')
    
# for i in range(len(replay_memory_50)):
#     camera_obs, proprio_obs, action, feedback=replay_memory_50[i]
#     replay_memory_75.camera_obs.append(camera_obs)
#     replay_memory_75.proprio_obs.append(proprio_obs)
#     replay_memory_75.action.append(action)
#     replay_memory_75.feedback.append(feedback)

# print(f'length of demos in 75 {len(replay_memory_75)}')

# for i in range(len(replay_memory_75)):
#     camera_obs, proprio_obs, action, feedback=replay_memory_75[i]
#     replay_memory_100.camera_obs.append(camera_obs)
#     replay_memory_100.proprio_obs.append(proprio_obs)
#     replay_memory_100.action.append(action)
#     replay_memory_100.feedback.append(feedback)

# print(f'length of demos in 100 {len(replay_memory_100)}')


# torch.save(replay_memory_50, "data/wire_bc/demos_50.dat")
# torch.save(replay_memory_75, "data/wire_bc/demos_75.dat")
# torch.save(replay_memory_100,"data/wire_bc/demos_100.dat")
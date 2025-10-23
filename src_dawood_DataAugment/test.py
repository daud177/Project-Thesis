import torch
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds
from models import Policy

config= {
        "feedback_type": 'cloning_10',
        "task": 'intro_test',
        "episodes": 50, #
        "static_env": False,
        "headless_env": False,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 8,#16
    }

replay_memory_10 = torch.load("data/intro_test/demos_10.dat")

model_path = "data/intro_test/cloning_10_policy.pt"

policy = Policy(config)

policy.load_state_dict(torch.load(model_path))

print(f'length of demos {len(replay_memory_10)}')
    
# replay_memory_30 = torch.load("data/stator_pick_100/intermediate_1end30.dat")
# replay_memory_43 = torch.load("data/stator_pick_100/intermediate_2end43.dat")
# replay_memory_12 = torch.load("data/stator_pick_100/intermediate_end12.dat")
# replay_memory_15 = torch.load("data/stator_pick_100/demos_15.dat")

# demos_100 = replay_memory_30

# print(f'length of demos {len(demos_100)}')

# for i in range(len(replay_memory_43)-2):
#     camera_obs, proprio_obs, action, feedback=replay_memory_43[i]
#     demos_100.camera_obs.append(camera_obs)
#     demos_100.proprio_obs.append(proprio_obs)
#     demos_100.action.append(action)
#     demos_100.feedback.append(feedback)

# print(f'length of demos {len(demos_100)}')
    

# for i in range(len(replay_memory_12)-1):
#     camera_obs, proprio_obs, action, feedback=replay_memory_12[i]
#     demos_100.camera_obs.append(camera_obs)
#     demos_100.proprio_obs.append(proprio_obs)
#     demos_100.action.append(action)
#     demos_100.feedback.append(feedback)

# print(f'length of demos {len(demos_100)}')

# for i in range(len(replay_memory_15)):
#     camera_obs, proprio_obs, action, feedback=replay_memory_15[i]
#     demos_100.camera_obs.append(camera_obs)
#     demos_100.proprio_obs.append(proprio_obs)
#     demos_100.action.append(action)
#     demos_100.feedback.append(feedback)

# print(f'length of demos {len(demos_100)}')

# torch.save(demos_100, "demos_100.dat")
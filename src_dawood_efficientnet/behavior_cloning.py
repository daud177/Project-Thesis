import wandb
import torch
import numpy as np
import optuna
from models_training import Policy
from utils import TrajectoriesDataset  # noqa: F401
from utils import set_seeds
from argparse import ArgumentParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset  # repeat the dataset
import os
from datetime import datetime
import random
import torch.distributed as dist
import os

def train_step(policy, step, replay_memory, config, len, rank):
    dataLoader_local = get_data_loader(replay_memory, batch_size=config["batch_size"], len_dataset=len)   # get dataloader in each train step
    for batch in dataLoader_local: # update parameters
        if step == 0:
            print(f'step {step + 1}')

        camera_batch, proprio_batch, action_batch, feedback_batch = batch
        training_metrics = policy.module.update_params(
            camera_batch, proprio_batch, action_batch, feedback_batch, rank
        )

        wandb.log(training_metrics)
        # print(f'step {step+1}')
        # print(training_metrics) # Check loss in every step

        if (step + 1) % 100 == 0:
            print(f'step {step + 1}')   # print overall steps
            print(training_metrics)     # print loss information

    return training_metrics["loss"]

def custom_collate_fn(batch):
    camera_batch = torch.stack([item[0] for item in batch], dim=1)
    proprio_batch = torch.stack([item[1] for item in batch], dim=1)
    action_batch = torch.stack([item[2] for item in batch], dim=1)
    feedback_batch = torch.stack([item[3] for item in batch], dim=1)
    return camera_batch, proprio_batch, action_batch, feedback_batch

def get_data_loader(dataset, batch_size, len_dataset):
    indices = list(range(len_dataset))  # create indices' list
    np.random.shuffle(indices)  # shuffle indices
    subset_indices = indices[:batch_size]  # pick front [batch_size] indices as sub indices
    print(subset_indices)
    sampler = SubsetRandomSampler(subset_indices)  # create sampler
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=custom_collate_fn)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(rank, config, world_size):

    print(torch.cuda.is_available())
    print("-------------------")
    if not torch.cuda.is_available():
        print("CUDA-capable GPU is not available.")
        return
    
    start_time = datetime.now()  # record start time
    print(f"Training started at: {start_time}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    wandb.init(config=config, project="ceiling", mode="online")
    if config["feedback_type"] == "pretraining":
        dataset_name = "/demos_10.dat"
        config["steps"] = 800  # old800
    elif config["feedback_type"] == "cloning_10":
        dataset_name = "/demos_10.dat"
        config["steps"] = 2000
    elif config["feedback_type"] == "cloning_25":  #
        dataset_name = "/demos_25.dat"
        config["steps"] = 2000
    elif config["feedback_type"] == "cloning_50":  #
        dataset_name = "/demos_50.dat"
        config["steps"] = 2000
    elif config["feedback_type"] == "cloning_75":  #
        dataset_name = "/demos_75.dat"
        config["steps"] = 2000
    elif config["feedback_type"] == "cloning_100":
        dataset_name = "/demos_100.dat"
        config["steps"] = 500
    elif config["feedback_type"] == "cloning_200":
        dataset_name = "/demos_200.dat"
        config["steps"] = 2000
    elif config["feedback_type"] == "cloning_500":
        dataset_name = "/demos_200.dat"
        config["steps"] = 5000    
    else:
        raise NotImplementedError

    replay_memory = torch.load("data/" + config["task"] + dataset_name)
    replay_memory = ConcatDataset([replay_memory] * 5)
    len_replay_memory = len(replay_memory)  # number of trajectories / episodes

    policy = Policy(config, rank, world_size).to(rank)
    policy = DDP(policy, device_ids=[rank])

    for i in range(config["steps"]):
        final_loss = train_step(policy, i, replay_memory, config, len=len_replay_memory, rank=rank)

    file_name = "data/" + config["task"] + "/" + config["feedback_type"] + f"_policy.pt"
    if rank == 0:
        torch.save(policy.module.state_dict(), file_name)
        wandb.watch(policy, log_freq=100)
    dist.destroy_process_group()

    end_time = datetime.now()  # record end time
    duration = end_time - start_time  # calculation duration
    print(f"Training ended at: {end_time}")
    print(f"Total training duration: {duration}")
    wandb.finish()
    return final_loss

def objective(trial):
    config = {
        "task": "Bowen_Vit",
        "feedback_type": "cloning_50",
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 224,
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "num_layers": trial.suggest_int("num_layers", 4, 6, step=1),
        "dim_feedforward": trial.suggest_int("dim_feedforward", 1024, 4096, step=1024),
        "steps": 100,
        "model_type": trial.suggest_categorical("model_type", ["resnet18", "densenet121"])
    }

    world_size = 2  # Number of GPUs
    results = spawn(main, args=(config, world_size), nprocs=world_size, join=True)
    
    # Assuming results contain the final training loss from all ranks
    # Use rank 0's loss as the objective value
    final_loss = results[0]
    return final_loss

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'  # or the appropriate IP address
    os.environ['MASTER_PORT'] = '12345'  # an available port
    
    set_seeds(1)

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning_10",
        help="options: pretraining, cloning_10, cloning_100",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        help="options: ApproachCableConnector, ApproachCableStrand, ApproachStator, GraspCableConnector, GraspStator, PickUpStatorReal",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize hyperparameters using Optuna"
    )
    args = parser.parse_args()

    if args.optimize:
        study_name = "hyperparameter_optimization"
        storage_name = "sqlite:///optuna_study.db"
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True)
        study.optimize(objective, n_trials=50)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        best_params = trial.params
        with open("best_params.txt", "w") as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
    else:
        config_defaults = {
            "feedback_type": args.feedback_type,
            "task": args.task,
            "proprio_dim": 8,
            "action_dim": 7,
            "visual_embedding_dim": 256,
            "learning_rate": 3e-4,
            "weight_decay": 3e-6,
            "batch_size": 1
        }
        world_size = 2  # Number of GPUs
        spawn(main, args=(config_defaults, world_size), nprocs=world_size, join=True)

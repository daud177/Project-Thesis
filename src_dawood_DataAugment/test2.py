import torch
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds


from models import Policy


import wandb
import torch
from models import Policy
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds
from argparse import ArgumentParser




def main(config):
    
    policy = Policy(config).to(device)
    
    pytorch_total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(pytorch_total_params)

    return



if __name__ == "__main__":
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
    args = parser.parse_args()

    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 8,#16
    }
    wandb.init(config=config_defaults, project="ceiling", mode="disabled")
    config = wandb.config  # important, in case the sweep gives different values
    main(config)

    

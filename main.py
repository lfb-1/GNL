# from train_cifar import Trainer
from train_cifar import CIFAR_Trainer
from train_red import RED_Trainer
from train_cifarN import CIFARN_Trainer
from configs import *
import argparse
import torch
import numpy as np

import random

parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
parser.add_argument("--name", type=str)
parser.add_argument("--r", default=0.5, type=float)
parser.add_argument("--root", default="/media/hdd/fb/cifar-10-batches-py", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--desc", default="baseline", type=str)
parser.add_argument("--config", default="cifar10", type=str)
parser.add_argument("--optim_goal", default="pxy", type=str)
parser.add_argument("--target", default="worse_label", type=str, help="Used only for CIFAR_N")
args = parser.parse_args()

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# for i in [0.2,0.3,0.4,0.5,0.6]:
print(f"Optimize in {args.optim_goal}")
if args.config == "cifar10":
    config = cifar10_configs(args.r, args.root, args.optim_goal)
    trainer = CIFAR_Trainer(config, args.desc)
elif args.config == "cifar100":
    config = cifar100_configs(args.r, args.root, args.optim_goal)
    trainer = CIFAR_Trainer(config, args.desc)
elif args.config == "red":
    config = red_configs(args.r, args.root, args.optim_goal)
    trainer = RED_Trainer(config, args.desc)
elif args.config == "cifar10n":
    config = cifar10n_configs(args.target, args.root, args.optim_goal)
    trainer = CIFARN_Trainer(config, args.desc)
elif args.config == "cifar100n":
    config = cifar100n_configs(args.target, args.root, args.optim_goal)
    trainer = CIFARN_Trainer(config, args.desc)
else:
    raise NotImplementedError
trainer.pipeline(trainer.train)

# config = cifar10_configs(args.r,args.root)
# trainer = CIFAR_Trainer(config,args.desc)
# trainer.pipeline(trainer.train)


# config = meta_cifar10_configs(args.r, args.root, args.clean_size)
# trainer = Trainer(config, args.desc)
# trainer.meta_pipeline()

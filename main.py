# from train_cifar import Trainer
from train_cifar import CIFAR_Trainer
from configs import * 
import argparse
import torch
import numpy as np

import random
parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
parser.add_argument("--name", type=str)
parser.add_argument("--r", default=0.5, type=float)
parser.add_argument("--root", default="/media/hdd/fb/cifar-10-batches-py", type=str)
parser.add_argument('--seed',default=123,type=int)
parser.add_argument('--desc',default='baseline',type=str)
args = parser.parse_args()

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

config = cifar10_configs(args.r,args.root)
trainer = CIFAR_Trainer(config,args.desc)
trainer.pipeline(trainer.train)


# config = meta_cifar10_configs(args.r, args.root, args.clean_size)
# trainer = Trainer(config, args.desc)
# trainer.meta_pipeline()

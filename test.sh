#!/bin/bash
#SBATCH --job-name=GNL    # Set a job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=6                    # Number of tasks (processes)
#SBATCH --mem=16G                     # Memory per node
#SBATCH --gres=gpu:nvidia_rtx_a6000:1               # Number of GPUs
#SBATCH --partition=sablab # Partition name
#SBATCH --time=120:00:00

source activate torch2
# conda run -n brats-submission nnUNetv2_predict -i /share/sablab/nfs04/users/hk672/brats/validation_data_nnunet_format/env-test -o . -d 1001 -p nnUNetPlans -tr nnUNetTrainerSegResNet -c 3d_fullres -f 0 --save_probabilities

# python train.py -a vgg16 --dist-url 'tcp://127.0.0.1:23488' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/datasets/fl453/ILSVRC2012/imagenet --workers 8 --lr 0.001 --pretrained --epochs 20 --desc multifc_ft
# python train.py -a resnet18 --dist-url 'tcp://127.0.0.1:23487' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/datasets/fl453/ILSVRC2012/imagenet  --workers 8  --lr 0.01 --pretrained --epochs 30 --desc ft
# python train.py -a resnet50 --dist-url 'tcp://127.0.0.1:23486' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /scratch/datasets/fl453/ILSVRC2012/imagenet  --workers 8  --lr 0.001 --pretrained --epochs 6 --desc multifc_afterfc
# rm -rf /scratch/datasets/fl453/ILSVRC2012/imagenet
# cp -r  /share/sablab/nfs04/data/ILSVRC2012/imagenet /scratch/datasets/fl453/ILSVRC2012/
# python generate_freeresponse_radio.py --start 0 --end 1000
./run.sh

# python generate_cardio.py

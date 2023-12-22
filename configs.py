from easydict import EasyDict


def cifar10_configs(r, root_dir, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 10
    config["total_epochs"] = 150
    config["lr"] = 0.02
    config["wd"] = 5e-4
    config["nesterov"] = False
    config["dataset"] = "cifar10"
    config["r"] = r
    config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["batch_size"] = 128
    config["num_workers"] = 5
    config["num_classes"] = 10
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [100]
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)


def cifar100_configs(r, root_dir, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 10
    config["total_epochs"] = 150
    config["lr"] = 0.02
    config["wd"] = 5e-4
    config["nesterov"] = False
    config["dataset"] = "cifar100"
    config["r"] = r
    config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["batch_size"] = 128
    config["num_workers"] = 5
    config["num_classes"] = 100
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [100]
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)


def red_configs(r, root_dir, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 10
    config["total_epochs"] = 150
    config["lr"] = 0.02
    config["wd"] = 5e-4
    config["nesterov"] = False
    config["dataset"] = "cifar100"
    config["r"] = r
    config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["batch_size"] = 128
    config["num_workers"] = 5
    config["num_classes"] = 100
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [100]
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)


def cifar10n_configs(target, root_dir, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 10
    config["total_epochs"] = 120 #300
    config["lr"] = 0.02
    config["wd"] = 5e-4
    config["nesterov"] = False
    config["dataset"] = "cifar10"
    # config["r"] = r
    config["target"] = target
    # config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["batch_size"] = 128
    config["num_workers"] = 5
    config["num_classes"] = 10
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [80] # 150/250
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)


def cifar100n_configs(target, root_dir, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 15
    config["total_epochs"] = 300
    config["lr"] = 0.02
    config["wd"] = 5e-4
    config["nesterov"] = False
    config["dataset"] = "cifar100"
    # config["r"] = r
    config["target"] = target
    # config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["batch_size"] = 128
    config["num_workers"] = 5
    config["num_classes"] = 100
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [150,250]
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)


def animal_configs(root_dir, cot=1, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 10
    config["total_epochs"] = 100
    config["lr"] = 0.02
    config["wd"] = 1e-3
    config["nesterov"] = False
    # config["dataset"] = "cifar100"
    # config["r"] = r
    # config['target'] = target
    # config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["cot"] = cot
    config["batch_size"] = 128
    config["num_workers"] = 5
    config["num_classes"] = 10
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [50]
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)

def c1m_configs(root_dir, optim_goal="pxy"):
    config = {}
    config["warmup_epochs"] = 1
    config["total_epochs"] = 80
    config["lr"] = 2e-3
    config["wd"] = 1e-3
    config["nesterov"] = False
    config["dataset"] = "cifar100"
    # config["r"] = r
    # config["target"] = target
    # config["noise_mode"] = "instance"
    config["root_dir"] = root_dir
    config["batch_size"] = 64
    config["num_workers"] = 5
    config["num_classes"] = 14
    config["wandb"] = True
    config["beta"] = 0.9
    config["lr_decay"] = [40]
    config["num_prior"] = 1
    config["optim_goal"] = optim_goal
    return EasyDict(config)
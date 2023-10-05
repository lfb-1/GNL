from easydict import EasyDict


def cifar10_configs(r, root_dir):
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
    config['num_prior'] = 1
    return EasyDict(config)

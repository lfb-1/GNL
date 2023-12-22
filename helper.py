class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class LogFT(object):
    def __init__(self, log_file) -> None:
        self.log_file = log_file

    def __call__(self, output) -> None:
        self.log_file.write(output)
        self.log_file.flush()
        print(output)

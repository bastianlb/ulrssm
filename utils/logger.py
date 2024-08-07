# logging utils for training and validation
import datetime
import logging
import time
from collections import defaultdict

from .dist_util import get_dist_info, master_only

# initialized logger
initialized_logger = {}


class AvgTimer:
    """
    Timer to record the average elapsed time.

    Usage:
        timer = AvgTimer()
        for _ in range(100):
            timer.start()
            ... # do something
            timer.record()
            print(timer.get_current_time()) # print current elapsed time
        print(timer.get_avg_time()) # print average elapsed time
    """

    def __init__(self, window=200):
        """
        Args:
            window (int, optional): Sliding window to compute average time. Default 200.
        """
        self.window = window
        self.current_time = 0.
        self.total_time = 0.
        self.avg_time = 0.
        self.count = 0
        self.start()

    def start(self):
        self.start_time = time.time()

    def record(self):
        self.count += 1
        # calculate current time
        self.current_time = time.time() - self.start_time
        # calculate total time
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset timer
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger:
    """
    Message Logger supporting both TensorBoard and Weights & Biases (optional)
    
    Args:
        opt (dict): Config dict containing:
            name (str): Experiment name.
            logger (dict): Contains 'print_freq' as logging interval.
            train (dict): Contains 'total_iter' as total iterations.
            use_tb_logger (bool): Whether to use TensorBoard logger.
            use_wandb (bool): Whether to use Weights & Biases logger.
        start_iter (int, optional): Start iteration number. Default 1.
        tb_logger (SummaryWriter, optional): TensorBoard logger. Default None.
    """
    def __init__(self, opt, start_iter=1):
        self.opt = opt
        self.exp_name = opt['name']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        logger = opt['logger'].get('type', "tensorboard")
        self.use_wandb = logger == "wandb"
        # use TensorBoard logger if wandb is not used
        self.use_tb_logger = logger == "tensorboard" or not self.use_wandb
        self.start_time = time.time()
        self.logger = get_root_logger()
        
        self.wandb = None
        if self.use_wandb:
            self._init_wandb(opt)

        if self.use_tb_logger:
            self.log_dir = opt['path']['experiments_root']
        
        self.log_buffer = defaultdict(list)

    @master_only
    def _init_wandb(self, opt):
        """Initialize wandb if it's being used."""
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=self.exp_name, config=opt)
        except ImportError:
            print("Weights & Biases (wandb) is not installed. Running without wandb logging.")
            self.use_wandb = False
            self.use_tb_logger = True

    @master_only
    def _init_tb_logger(self):
        from torch.utils.tensorboard import SummaryWriter
        self.tb_logger = SummaryWriter(log_dir=self.log_dir)

    def reset_start_time(self):
        """Reset start time."""
        self.start_time = time.time()

    def add_to_buffer(self, log_dict):
        """Add log_dict to buffer."""
        for k, v in log_dict.items():
            self.log_buffer[k].append(v)

    def flush_buffer(self, current_iter):
        """Flush buffer and log to TensorBoard and W&B."""
        for k, v in self.log_buffer.items():
            if len(v) > 0:
                avg_value = sum(v) / len(v)
                if self.use_tb_logger:
                    self.tb_logger.add_scalar(k, avg_value, current_iter)
                if self.use_wandb:
                    self.wandb.log({k: avg_value}, step=current_iter)
        self.log_buffer.clear()

    @master_only
    def __call__(self, log_dict):
        """
        Logging message
        
        Args:
            log_dict (dict): Logging dictionary with keys:
                epoch (int): Current epoch.
                iter (int): Current iteration.
                lrs (list): List of learning rates.
                time (float): Elapsed time for one iteration.
                data_time (float): Elapsed time of data fetch for one iteration.
        """
        epoch = log_dict.pop('epoch')
        current_iter = log_dict.pop('iter')
        lrs = log_dict.pop('lrs')

        message = f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:('
        message += ','.join([f'{lr:.3e}' for lr in lrs]) + ')]'

        if 'time' in log_dict:
            iter_time = log_dict.pop('time')
            data_time = log_dict.pop('data_time')
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, time (data): {iter_time:.3f} ({data_time:.3f})]'

        for k, v in log_dict.items():
            message += f'{k}: {v:.4e} '
            self.add_to_buffer({k: v})

        self.logger.info(message)
        
        if current_iter % self.opt['logger']['print_freq'] == 0:
            self.flush_buffer(current_iter)

    def close(self):
        """Close loggers."""
        if self.use_tb_logger:
            self.tb_logger.close()
        if self.use_wandb:
            self.wandb.finish()



def get_root_logger(logger_name='root_logger', log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str, optional): root logger name. Default: 'root_logger'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger. Default None.
        log_level (int, optional): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time. Default logging.INFO.

    Returns:
        logging.Logger: The root logger.
    """ 
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it.
    if logger_name in initialized_logger:
        return logger

    # initialize stream handler
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(stream_handler)
    logger.propagate = False

    # initialize logger level for each process
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        if log_file and not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    # add logger to initialized logger
    initialized_logger[logger_name] = True

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision
    import platform

    msg = ('\nVersion Information: '
           f'\n\tPython: {platform.python_version()}'
           f'\n\tPyTorch: {torch.__version__}'
           f'\n\tTorchVision: {torchvision.__version__}')
    return msg

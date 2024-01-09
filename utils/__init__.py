import os
import torch
import torch.distributed as dist
from utils.checkpoint_process import *
from collections import defaultdict, deque, OrderedDict
import time
import yaml
import datetime
from datetime import datetime as date_time
import numpy as np
import json
import torch
import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

##print configs
def print_namespace_as_table(namespace):
    # Extracting the attributes and values from the Namespace
    items = vars(namespace).items()

    # Finding the longest attribute name for formatting
    max_key_length = max(len(key) for key, _ in items)

    # Printing the table
    print("Key" + " " * (max_key_length - len("Key") + 2) + "| Value")
    print("-" * (max_key_length + 2) + "+" + "-" * 30)  # Adjust 30 based on expected value length

    for key, value in items:
        padding = " " * (max_key_length - len(key)+1)
        print(f"{key}{padding} | {value}")
    print('\n\n')

class ScstRewardCriterion(torch.nn.Module):
    CIDER_REWARD_WEIGHT = 1

    def __init__(self, cider_cached_tokens='corpus', baseline_type='greedy'):
        self.CiderD_scorer = CiderD(df=cider_cached_tokens)
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        super().__init__()

    def forward(self, gt_res, greedy_res, sample_res, sample_logprobs):
        batch_size = len(gt_res)
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()

        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

    def get_score(self):
        return self._cur_score

    def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [self._wrap_sentence(gen_res[i])]

        gts = OrderedDict()
        gt_res_ = [
            [self._wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))]
                for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[gt_idx[i]]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, batch_cider_scores = self.CiderD_scorer.compute_score(gts, res_)
        scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        return scores

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        r += ' <eos>'
        return r


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.8f} ({global_avg:.8f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.8f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, dataset_len=None, epoch_info=None):
        if not header:
            header = ''
        if not dataset_len:
            dataset_len = len(iterable)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.8f}')
        data_time = SmoothedValue(fmt='{avg:.8f}')
        space_fmt = ':' + str(len(str(dataset_len))) + 'd'

        _msg = [
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            _msg.append('max mem: {memory:.0f}')
        _msg = self.delimiter.join(_msg)
        MB = 1024.0 * 1024.0
        iterable = iter(iterable)
        train_steps = dataset_len
        if epoch_info:
            start_epoch, end_epoch = epoch_info
            train_steps = (end_epoch - start_epoch) * dataset_len
        for i in range(train_steps):
            obj = next(iterable)
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if epoch_info:
                header = int(i / dataset_len) + start_epoch
                header = 'Train step: [{}]'.format(header)
            log_msg = header + " " + _msg
            if (i % dataset_len) % print_freq == 0 or i == dataset_len - 1:
                eta_seconds = iter_time.global_avg * (dataset_len - i % dataset_len)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.8f} s / it)'.format(
            header, total_time_str, total_time / dataset_len))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)





def load_configuration(model_name, dataset_name):
    """Load configuration files for the model and dataset."""
    model_config = yaml.safe_load(open(f'./configs/models/{model_name}.yaml'))
    data_config = yaml.safe_load(open(f'./configs/data/{dataset_name}.yaml'))
    training_config = yaml.safe_load(open(f'./configs/training.yaml'))
    return model_config, data_config, training_config

def setup_environment(args):
    """Setup the training environment including distributed mode, seeds, and device."""
    init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = get_world_size()  # Retrieve the world size for distributed training
    set_random_seeds(42 + get_rank())
    cudnn.benchmark = True
    return device, world_size

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

        
def initialize_wandb(args):
    """Initialize Weights & Biases tracking."""
    if args.debug:
        tags=[args.model, args.task, args.dataset, args.version, "debug"]
    else:
        tags=[args.model, args.task, args.dataset, args.version]
    current_time = date_time.now().strftime("%d/%m/%y %H:%M")
    if is_main_process():
        branch = f"{args.model}/{args.task}/{args.version}/{args.dataset}"
        message = args.note
        os.system(f'git config --global user.email "dunghuynh110496@gmail.com"')
        os.system(f'git config --global user.name "parkerhuynh"')
        
        branch_exists = os.system(f"git rev-parse --verify {branch}")
        if branch_exists != 0:
            os.system(f"git branch {branch}")
        os.system(f"git checkout {branch}")
        os.system(f"git commit -am \"{message}\"")
        os.system(f"git remote --set-url origin https://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@github.com/parkerhuynh/HieVQA")
        code_version = os.popen('git rev-parse HEAD').read().strip()
        
        
        
        
        args.created = current_time
        args.branch = branch
        args.code_version = code_version
        args.commit_link =  f"https://github.com/parkerhuynh/HieVQA/commit/{code_version}"
        args.checkout = f"git checkout {code_version}"
        
        
        output_path = os.path.join(args.output_dir, f"{args.model}-{args.version}")
        if not os.path.exists(output_path):
            print(f"{'The output path doesnt exist:'.upper()} {output_path}")
            print('creating'.upper())
            os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path,  f'{args.dataset}-{args.task}-{code_version}')
        os.makedirs(output_path, exist_ok=True)
        args.output_dir = output_path
        
        wandb_group_name = f"{args.model}-{args.task}-{args.dataset}-{args.version}-{args.note}"
        wandb_running_name = f"cuda{get_rank()}-{current_time}-{args.model}-{args.version}-{args.task}-{args.dataset}"
        
        wandb.init(
            project="VQA",
            group=wandb_group_name,
            name= wandb_running_name,
            tags=tags,
            notes=args.note,
            config=vars(args),
            dir=args.wandb_dir
        )
    else:
        wandb_group_name = f"{args.model}-{args.task}-{args.dataset}-{args.version}-{args.note}"
        wandb_running_name = f"cuda{get_rank()}-{current_time}-{args.model}-{args.version}-{args.task}-{args.dataset}"
        wandb.init(
            project="VQA",
            group=wandb_group_name,
            name= wandb_running_name,
            tags=tags,
            notes=args.note,
            config=vars(args),
            dir=args.wandb_dir)
        
        
def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)
    
def list_files_and_subdirectories(path,args):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:

            if (".py" in file ) or (".json" in file) or (".yaml" in file) or (".txt" in file) or (".csv" in file):
                file_list.append(os.path.join(root, file))
    filtered_list = []
    for file in file_list:
        if "wandb" in file or '__pycache__' in file or 'cider' in file:
            continue
        elif 'result' in file:
            if args.output_dir not in file:
                continue
            filtered_list.append(file)
        else:
            filtered_list.append(file)
    return filtered_list
import os
from typing import Union, Dict, List, Tuple, Any, Callable
import logging
import re
import time
import torch

from utils.hdfs_io import hexists, hmkdir, hcopy
from utils.torch_io import save as hdfs_torch_save
logger = logging.getLogger(__name__)

def output_path_create(args):
    output_path = os.path.join(args.output_dir, args.model, args.task, args.dataset)
    if not os.path.exists(output_path):
        print(f"{'The output path doesnt exist:'.upper()} {output_path}")
        print('creating'.upper())
        os.makedirs(output_path, exist_ok=True)
    subfolder_count = len([name for name in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, name))])
    output_path = os.path.join(output_path,  f'version_{subfolder_count}')
    os.makedirs(output_path, exist_ok=True)
    return output_path

class Checkpointer:
    def __init__(self,
                 serialization_dir: str = ".output") -> None:
        self._serialization_dir = serialization_dir
        if not hexists(self._serialization_dir):
            hmkdir(self._serialization_dir)

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        step: int = -1) -> None:
        """
        Save ckpt to local or HDFS
        """
        if step > 0:
            model_path = os.path.join(
                self._serialization_dir, "model_state_step_{}.th".format(step))
            hdfs_torch_save(model_state, model_path)

        else:
            model_path = os.path.join(
                self._serialization_dir, "model_state_best.th".format(epoch))

            training_path = os.path.join(self._serialization_dir,
                                         "training_state_latest.th")
            hdfs_torch_save(model_state, model_path)
            hdfs_torch_save({**training_states, "epoch": epoch}, training_path)

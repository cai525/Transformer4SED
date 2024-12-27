from typing import Union
import yaml

import torch.nn as nn

from src.utils.statistics.model_statistic import count_parameters
from src.utils.scheduler import ExponentialDown, ExponentialWarmup, CosineDown, update_ema
from src.utils.log import BestModels, Logger


class DataParallelWrapper:

    def __init__(self, model: Union[nn.Module, nn.DataParallel]):
        self.model = model

    def __getattr__(self, name):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return getattr(model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_yaml_with_relative_ref(yaml_path) -> dict:
    """" 
        Load the YAML file that references another YAML file
    """
    with open(yaml_path, "r") as f:
        main_content = yaml.safe_load(f)
    if isinstance(main_content, dict) and "include" in main_content:
        included_content = main_content.pop("include")
        key_list = included_content['keys']
        assert included_content['base_path'] != yaml_path
        base_dict = load_yaml_with_relative_ref(included_content['base_path'])
        for key in key_list:
            main_content[key] = base_dict[key]
    return main_content

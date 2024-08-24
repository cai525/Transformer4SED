import torch.nn.functional as f

from recipes.finetune.train import Trainer
from src.models.cnn_transformer.passt_cnn import PaSST_CNN


class PaSST_CNN_Trainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)

    def validation(self, epoch):
        ret = super().validation(epoch)
        assert isinstance(self.net.module, PaSST_CNN)
        self.my_logger.logger.info("merge weight = {0}".format(self.net.module.merge_weight))
        return ret

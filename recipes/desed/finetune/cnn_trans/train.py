from recipes.desed.finetune.train import Trainer


class PaSST_CNN_Trainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)

    def validation(self, epoch):
        ret = super().validation(epoch)
        self.my_logger.logger.info("merge weight = {0}".format(self.net.merge_weight))
        return ret

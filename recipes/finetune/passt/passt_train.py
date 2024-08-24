from recipes.finetune.train import Trainer


class PasstTrainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)

from tqdm import tqdm

import torch

from recipes.desed.mlm.train import Trainer
from src.functional.loss import MSELoss
from src.preprocess.data_aug import frame_shift, feature_transformation


class MLMTrainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)
        self.reconstruction_loss = MSELoss()

    def train(self, epoch):
        self.net.train()
        n_train = len(self.train_loader)
        tk0 = tqdm(self.train_loader, total=n_train, leave=False, desc="training processing")

        mean_loss = 0

        for _, (wavs, _, _, _) in enumerate(tk0, 0):
            wavs = wavs.to(self.device)
            # Transform to mel respectively
            extractor = self.net.get_feature_extractor()
            mel = extractor(wavs)
            # normalization
            mel = extractor.normalize(mel)
            # time shift
            mel = frame_shift(mel)
            mel = feature_transformation(mel, log=True, norm_std=5.0, **self.config["training"]["transform"])

            pred, other_dict = self.net(mel, encoder_win=self.config["training"]["encoder_win"])
            assert pred.shape[1] == 1000
            frame_before_mask = other_dict["frame_before_mask"]
            mask_id_seq = other_dict["mask_id_seq"]
            loss = self.reconstruction_loss(frame_before_mask[mask_id_seq], pred[mask_id_seq])

            torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            mean_loss += loss.item() / n_train

        self.logger.info("Epoch {0}: Train loss is {1}".format(epoch, mean_loss))
        self.logger.info("Epoch {0}: lr scale is {1}".format(epoch, self.scheduler._get_scale()))
        return

    def validation(self, epoch):
        self.net.eval()
        n_valid = len(self.val_loader)
        mean_loss = 0

        with torch.no_grad():
            tk1 = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for _, (wavs, _, _, _) in enumerate(tk1, 0):
                wavs = wavs.to(self.device)
                # Transform to mel respectively
                extractor = self.net.get_feature_extractor()
                mel = extractor(wavs)
                # normalization
                mel = extractor.normalize(mel)
                pred, other_dict = self.net(mel, encoder_win=self.config["training"]["encoder_win"])
                frame_before_mask = other_dict["frame_before_mask"]
                mask_id_seq = other_dict["mask_id_seq"]
                loss = self.reconstruction_loss(frame_before_mask[mask_id_seq], pred[mask_id_seq])
                mean_loss += loss.item() / n_valid

        self.logger.info("Epoch {0}: Validation reconstruction loss is {1:.4f}".format(epoch, loss))
        return mean_loss

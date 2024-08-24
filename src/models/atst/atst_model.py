import logging

import torch

from src.models.atst.audio_transformer import FrameASTModel


class ATST(torch.nn.Module):

    def __init__(
        self,
        atst_path,
        *args,
        atst_dropout=0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.atst = FrameASTModel(atst_dropout=atst_dropout)
        self.load_atst(atst_path)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def set_cls_embed(self, cls_embed):
        self.cls_embed = cls_embed

    def forward(self, atst_feat: torch.Tensor, other_emb=None, n=1):
        atst_feat = atst_feat.unsqueeze(1)
        atst_dict = self.atst.get_intermediate_layers(
            atst_feat,
            self.fake_length.to(atst_feat).repeat(len(atst_feat)),
            n,
            scene=False,
            other_emb=other_emb,
        )
        return atst_dict

    def load_atst(self, path=None):
        if path is None:
            logging.info("<INFO> Don't load the pretrained ATST weight.")
            return
        logging.info("<INFO> Loading ATST from {0}".format(path))
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        atst_state_dict = {}
        for k, v in state_dict.items():
            if "model.teacher.encoder." in k:
                if "encoder.norm." in k:
                    new_k = k.replace("model.teacher.encoder.norm", "norm_frame")
                elif "cls_token" in k:
                    continue
                else:
                    new_k = k.replace("model.teacher.encoder.", "")
                atst_state_dict[new_k] = v
            # C2F
            if "encoder.encoder.frame_encoder." in k:
                new_k = k.replace("encoder.encoder.frame_encoder.", "")
                atst_state_dict[new_k] = v
                continue
            if "encoder.encoder.teacher_module." in k:
                continue
            # ATST-Frame
            if "encoder.encoder." in k:
                new_k = k.replace("encoder.encoder.", "")
                atst_state_dict[new_k] = v

        self.atst.load_state_dict(atst_state_dict, strict=True)

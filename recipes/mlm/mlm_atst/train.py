import random

import torch
from tqdm import tqdm

from src.models.atst.atst_sed import AtstSED
from src.preprocess.data_aug import frame_shift, feature_transformation

def train(train_cfg):
    net = train_cfg["net"]
    assert isinstance(net, torch.nn.parallel.DataParallel)
    assert isinstance(net.module, AtstSED)
    net.train()
    n_train = len(train_cfg["trainloader"])
    tk0 = tqdm(train_cfg["trainloader"], total=n_train, leave=False, desc="training processing")
    
    mean_loss = 0
    
    for _, (wavs, _, _, _) in enumerate(tk0, 0):
        wavs = wavs.to(train_cfg["device"])
        # Data preprocessing
        mel= net.module.get_feature_extractor().forward(wavs)
        
        # time shift
        mel = frame_shift(mel)
        mel, _ = feature_transformation(mel, **train_cfg["transform"])
        
        pred, other_dict = net(mel, encoder_win=train_cfg["encoder_win"])
        frame_before_mask = other_dict["frame_before_mask"]
        mask_id_seq = other_dict["mask_id_seq"] 
        loss = train_cfg["criterion_cons"](frame_before_mask[mask_id_seq], pred[mask_id_seq])

        torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=20, norm_type=2)
        loss.backward()
        train_cfg["optimizer"].step()
        train_cfg["optimizer"].zero_grad()
        train_cfg["scheduler"].step()
        mean_loss += loss.item()/n_train

    train_cfg['logger'].info("Epoch {0}: Train loss is {1}".format(train_cfg["epoch"], mean_loss))
    train_cfg['logger'].info("Epoch {0}: lr scale is {1}".format(train_cfg["epoch"], train_cfg["scheduler"]._get_scale()))
    return 
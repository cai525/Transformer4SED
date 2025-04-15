import torch
from tqdm import tqdm

from src.preprocess.preprocess import preprocess_eval

def validation(train_cfg):
    train_cfg["net"].eval()
    n_valid = len(train_cfg["validloader"])
    mean_loss = 0
    
    with torch.no_grad():
        tk1 = tqdm(train_cfg["validloader"], total=n_valid, leave=False, desc="validation processing")
        for _, (wavs, _, _, _) in enumerate(tk1, 0):
            wavs = wavs.to(train_cfg["device"])
            pred, other_dict = train_cfg["net"](preprocess_eval(wavs), encoder_win=train_cfg["encoder_win"])
            frame_before_mask = other_dict["frame_before_mask"]
            mask_id_seq = other_dict["mask_id_seq"] 
            loss = train_cfg["criterion_cons"](frame_before_mask[mask_id_seq], pred[mask_id_seq])
            mean_loss += loss.item()/n_valid
            
    train_cfg['logger'].info("Epoch {0}: Validation loss is {1}".format(train_cfg["epoch"], mean_loss))

    return loss
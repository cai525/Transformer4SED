import torch
import torch.nn.functional as F


class PasstWithSlide(torch.nn.Module):
    def __init__(self, net, win_param=[512, 29]):   
        super().__init__()     
        self.emb_len = 1000
        self.net = net
        self.out_dim = self.net.out_dim
        self.win = win_param


    def forward(self, input:torch.Tensor):
        """
        Returns:
            (sed_out, weak_out)
        """
        device = input.device
        batch_size, _, input_len = input.shape
        scale = self.emb_len / input_len

        # decode output shape :[batch, frame, shape]
        # For 10 seconds audio, frame length is 101
        embedding = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        accumlator =  torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        win_width, step = self.win
        res = []
        embedding = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        accumlator = torch.zeros([batch_size, self.emb_len, self.out_dim]).to(device)
        
        for w_left in range(0, input_len + step - win_width, step):
            w_right = min(w_left + win_width, input_len)
            out_left = round(w_left * scale)
            out = self.encode(input[:, :, w_left:w_right])
            out_right = int(min(self.emb_len, out_left + out.shape[1]))
            embedding[:, out_left:out_right, :] += out
            accumlator[:, out_left:out_right, :] += 1

        embedding /= accumlator
        embedding[torch.isnan(embedding)] = 0
        res.append(embedding)
            
        embedding = sum(res)/len(res)
        return embedding


    def encode(self, input: torch.Tensor):
        #patch-wise context modeling
        input = input.unsqueeze(1)
        passt_out_dict = self.net.patch_transformer(input)
        passt_feature = passt_out_dict["layer{}_out".format(self.net.passt_feature_layer)]
        passt_feature = passt_feature.transpose(1, 2)  # N,C,P->N,P,C
        passt_feature = self.net.out_norm(passt_feature)
        
        B_, P_, C_ = passt_feature.shape
        # print(encoder_out['f_dim'])j
        # print(encoder_out['t_dim'])
        passt_feature = passt_feature[:, 2:, :].reshape(B_, passt_out_dict['f_dim'], passt_out_dict['t_dim'], C_)
        frameout = torch.mean(passt_feature, dim=1)

        assert self.net.decode_ratio != 1
        frameout = frameout.transpose(1, 2)  # B,T,C->B,C,T
        frameout = F.interpolate(frameout, scale_factor=self.net.decode_ratio, mode=self.net.interpolate_module.mode)
        frameout = frameout.transpose(1, 2)  # B,C,T->B,T,C
        return frameout

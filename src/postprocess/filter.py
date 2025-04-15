import torch

def MyMedianfilterfunc(x,median_filter):
    ''' input dim Batch,T_dim,class_dim(10) '''

    Batch, T_size = x.shape[0], x.shape[1]
    out = []
    for class_idx in range(10):
        x_i = x[:, :, class_idx].unsqueeze(1).unsqueeze(-1)#(Batch*1*Tdim*1)
        median_filter_size_i = median_filter[class_idx]
        # print(cfg.classes[class_idx], median_filter_size_i)
        median_filter_size_i = median_filter_size_i + 1 if median_filter_size_i % 2 == 0 else median_filter_size_i#change to odd number
        pad_size_i = (0, 0, int(median_filter_size_i / 2), int(median_filter_size_i / 2))
        x_i = torch.nn.functional.pad(x_i, pad_size_i, mode='replicate')
        x_i = x_i.unfold(2, median_filter_size_i, 1).unfold(3, 1, 1)
        #print("!!!!!test!!!!", x_i.size())
        #print("!!!!!test!!!!", x_i.size()[:4])
        #print("!!!!!test!!!!",x_i.size()[:4] + (-1,))
        x_i = x_i.contiguous().view(x_i.size()[:4] + (-1,)).median(dim=-1)[0]
        out.append(x_i)

    out = torch.cat(out, dim=3).squeeze(1)
    return out

def MyMedianfilterfunc_tiny(x,median_filter_size):
    ''' input dim (T_dim,)'''

    x = x.unsqueeze(0).unsqueeze(1).unsqueeze(-1)#1,1,T_dim,1
    median_filter_size = median_filter_size + 1 if median_filter_size % 2 == 0 else median_filter_size#change to odd number
    pad_size = (0, 0, int(median_filter_size / 2), int(median_filter_size / 2))
    x = torch.nn.functional.pad(x, pad_size, mode='replicate')
    x = x.unfold(2, median_filter_size, 1).unfold(3, 1, 1)
    #print("!!!!!test!!!!", x_i.size())
    #print("!!!!!test!!!!", x_i.size()[:4])
    #print("!!!!!test!!!!",x_i.size()[:4] + (-1,))
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

    return x.squeeze(0).squeeze(0).squeeze(-1)
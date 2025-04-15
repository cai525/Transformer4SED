"""
This module aims to add slide window opreation in the prediction process.

It works without affecting the structure of previous version.

The module works to this location : 

```python
stud_preds, weak_stud_preds,other_dict_stud = train_cfg["net"](logmels)
```
"""
import torch
import torch.nn as nn


class SlideWindow:

    def __init__(self, win_width, step, n_class) -> None:
        self.win_width = win_width
        self.step = step
        self.n_class = n_class

    def __call__(self, fun, input: torch.Tensor, scale=1.0):
        """ 
            Act as the decorator of function "fun". Add slide window opreation 
            in the prediction process, and concatenate the predictions of 
            all the windows to a whole prediction.
        Args:
            fun : function used to make prediction 
            input (torch.Tensor): Input of function.
            scale (float, optional): The scale of output comparing to input.\
            Defaults to 1.0.

        Returns:
            Prediction of input concatenated with predictions of all the windows.
        """
        # input size : [bs, freqs, frames]
        # labels size = [bs, n_class, frames]
        device = input.device
        batch_size, _, input_len = input.shape
        output_len = int(input_len * scale)
        accumlator = torch.zeros([batch_size, self.n_class, output_len]).to(device)
        output = torch.zeros([batch_size, self.n_class, output_len]).to(device)

        for w_left in range(0, input_len + self.step - self.win_width, self.step):
            w_right = min(w_left + self.win_width, input_len)
            out_left, out_right = int(w_left * scale), int(min(w_right * scale, output_len))
            strong = fun(input[:, :, w_left:w_right])
            output[:, :, out_left:out_right] += strong[:, :, :out_right - out_left]
            accumlator[:, :, out_left:out_right] += 1

        output /= accumlator

        weak_out = (output * output).sum(dim=2) / output.sum(dim=2)
        weak_out = torch.clamp(weak_out, 1e-7, 1.)

        return output, weak_out

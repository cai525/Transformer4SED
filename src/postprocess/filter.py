import torch


def median_filter_torch(input_tensor, filter_size: list):
    """
    Apply a median filter to a 1D tensor along the Length dimension for each class in the batch.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape (Batch, Length, Classes).
        median_filter_sizes (list): List of median filter sizes, one for each class.

    Returns:
        torch.Tensor: Tensor with the same shape as input_tensor, after median filtering.
    """
    # Validate input dimensions
    if len(input_tensor.shape) != 3:
        raise ValueError("input_tensor must have shape (Batch, Length, Classes)")

    batch, length, num_classes = input_tensor.shape

    # Validate filter size list
    if len(filter_size) != num_classes:
        raise ValueError("Length of median_filter_sizes must match the number of classes")
    out = torch.zeros_like(input_tensor)
    for class_idx in range(10):
        x_i = input_tensor[:, :, class_idx].unsqueeze(1).unsqueeze(-1)  #(batch, 1, length, 1)
        # Get the filter size and adjust it to be odd
        filter_size_i = filter_size[class_idx]
        filter_size_i = filter_size_i + 1 if filter_size_i % 2 == 0 else filter_size_i  #change to odd number
        # Pad the tensor
        pad_size_i = (0, 0, int(filter_size_i / 2), int(filter_size_i / 2))
        x_i = torch.nn.functional.pad(x_i, pad_size_i, mode='replicate')
        x_i = x_i.unfold(2, filter_size_i, 1).unfold(3, 1, 1)
        x_i = x_i.contiguous().view(x_i.size()[:4] + (-1, )).median(dim=-1)[0]
        out[:, :, class_idx] = x_i.view(batch, length)
    return out

"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .filter import relevance_filter

angle = 180 # 2°
top_k_percent = 1.0 #0.5  # Proportion of relevance scores that are allowed to pass.
theta = angle * math.pi / 180 
sin_theta = math.sin(theta)
cos_theta = math.cos(theta)


def m_rule(w, x, r, eps=1e-6):
    bs, in_dim = x.size()
    n_filters = w.size()[0]
    normalized_w = w / (w.norm(dim=-1, keepdim=True) + eps)
    norm_x = x.norm(dim=-1, keepdim=True) + eps
    normalized_x = x / norm_x
    
    # 计算与x正交的单位正交向量
    wx = normalized_x.matmul(normalized_w.T) # [B, out_dim]
    if (wx.abs() == 1).any():
        print("orth exist.")
    orth = wx.view(bs, n_filters, 1) * normalized_x.view(bs, 1, in_dim) # [bs, n_filter, in_dim]
    orth = normalized_w - orth 
    
    normalized_orth = orth / (orth.norm(dim=-1, keepdim=True) + eps)
    
    x_head = cos_theta * normalized_x.view(bs, 1, in_dim) + sin_theta * normalized_orth
    x_head = x_head * norm_x.view(bs, 1, 1)
    x_diff = x.view(bs, 1, in_dim) - x_head
    # x_diff = x_diff / (x.view(bs, 1, in_dim) + eps)

    wx_diff = w * x_diff
    wx_diff_normaliezd = wx_diff / (wx_diff.sum(-1, keepdim=True) + eps)
    r = wx_diff_normaliezd * r.view(bs, n_filters, 1)
    r = r.sum(1, keepdim=False).data
    return r

class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AdaptiveAvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    # def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
    #     z = self.layer.forward(a) + self.eps
    #     s = (r / z).data
    #     (z * s).sum().backward()
    #     c = a.grad
    #     r = (a * c).data
    #     return r
    def forward(self, x: torch.tensor, r: torch.tensor) -> torch.tensor:
        if x.size() == r.size():
            return r.data

        batch_size, channel, height, width = x.size()
        out_h, out_w = r.size()[-2:]
        # print(x.size(), r.size())
        relevance = torch.zeros_like(x)
        # output = torch.zeros(batch_size, channel, out_h, out_w)
        for i in range(out_h):
            sh = math.floor(i * height / out_h) # start h index
            eh = math.ceil((i + 1) * height / out_h) # end h index
            for j in range(out_w):
                sw = math.floor(j * width / out_w) # start w index
                ew = math.ceil((j + 1) * width / out_w) # end w index
                
                # output[:, :, i:i+1, j:j+1] = torch.mean(x[:, :, sh: eh, sw: ew], dim=(-2, -1), keepdim=True)
                kernel_size = (eh - sh) * (ew - sw)
                x_col = x[:, :, sh: eh, sw: ew].view(batch_size*channel, kernel_size)
                w_col = torch.ones(1, kernel_size) / kernel_size
                r_col = r[:, :, i:i+1, j:j+1].view(-1, 1)
                r_col = m_rule(w_col, x_col, r_col, self.eps)
                relevance[:, :, sh: eh, sw: ew] += r_col.view(batch_size, channel, eh-sh, ew-sw).data
        # print(relevance.data)
        return relevance.data


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps

    # def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
    #     z = self.layer.forward(a) + self.eps
    #     s = (r / z).data
    #     (z * s).sum().backward()
    #     c = a.grad
    #     r = (a * c).data
    #     return r
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        kernel_size = self.layer.kernel_size
        stride = self.layer.stride
        padding = self.layer.padding
        # print(kernel_size, stride, padding)

        batch_size, channel, height, width = x.size()
        filter_h, filter_w = kernel_size
        out_h = int((height + padding[0]*2 - filter_h) / stride[0] + 1)
        out_w = int((width + padding[1]*2 - filter_w) / stride[1] + 1)
        x_col = F.unfold(x.view(batch_size * channel, 1, height, width), 
                         (filter_h, filter_w), 
                         padding=padding, 
                         stride=stride, 
                         dilation=dilation).transpose(1, 2).view(-1, filter_w*filter_h)
        # [batch_size * out_h*out*w, filter_w*filter_h*channel]
        # w_col = weight.view(n_filter, -1)
        w_col = torch.ones(1, filter_h * filter_w)
        r_col = r.view(-1, 1)
        r = m_rule(w_col, x_col, r_col, self.eps)
        r = F.fold(r.view(batch_size*channel, out_h*out_w, filter_w*filter_h).transpose(1, 2), 
                   output_size=(height, width), 
                   kernel_size=(filter_h, filter_w), 
                   padding=padding, 
                   stride=stride)
        return r.data


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.MaxPool2d, mode: str = "max", eps: float = 1.0e-05) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps

    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r



class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Conv2d, mode: str = "nz_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps

    # def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
    #     r = relevance_filter(r, top_k_percent=top_k_percent)
    #     z = self.layer.forward(a) + self.eps
    #     s = (r / z).data
    #     (z * s).sum().backward()
    #     c = a.grad
    #     r = (a * c).data
    #     return r
    def forward(self, x: torch.tensor, r: torch.tensor) -> torch.tensor:
        weight = self.layer.weight
        bias = self.layer.bias
        stride = self.layer.stride
        padding = self.layer.padding
        dilation = self.layer.dilation
        groups = self.layer.groups
        
        batch_size, channel, height, width = x.size()
        n_filter, _, filter_h, filter_w = weight.size()
        out_h = int((height + padding[0]*2 - dilation[0] * (filter_h - 1) - 1) / stride[0] + 1)
        out_w = int((width + padding[1]*2 - dilation[1] * (filter_w - 1) - 1) / stride[1] + 1)
        x_col = F.unfold(x, 
                         (filter_h, filter_w), 
                         padding=padding, 
                         stride=stride, 
                         dilation=dilation).transpose(1, 2).view(-1, filter_w*filter_h*channel)
        # [batch_size * out_h*out*w, filter_w*filter_h*channel]
        w_col = weight.view(n_filter, -1)
        # [n_filter, filter_w*filter*h*channel]
        # out = x_col.transpose(1, 2).matmul(weight.view(n_filter, -1).t()) + bias # [batch, out_img_size, n_filter]
        # out = out.transpose(1, 2).view(batch_size, n_filter, out_h, out_w)
        r_col = r.permute(0, 2, 3, 1).view(batch_size*out_h*out_w, n_filter)
        r = m_rule(w_col, x_col, r_col, self.eps)
        r = F.fold(r.view(batch_size, out_h*out_w, filter_w*filter_h*channel).transpose(1, 2), 
                   output_size=(height, width), 
                   kernel_size=(filter_h, filter_w), 
                   padding=padding, 
                   stride=stride,
                   dilation=dilation)
        return r.data


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(self, layer: torch.nn.Linear, mode: str = "nz_plus", eps: float = 1.0e-05) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            # self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps

    # @torch.no_grad()
    # def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
    #     r = relevance_filter(r, top_k_percent=top_k_percent)
    #     z = self.layer.forward(a) + self.eps
    #     s = r / z
    #     c = torch.mm(s, self.layer.weight)
    #     r = (a * c).data
    #     return r
    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = relevance_filter(r, top_k_percent=top_k_percent)
        r = m_rule(self.layer.weight, a, r, self.eps)
        return r.data


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r
        

class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r

class RelevancePropagationBatchNorm2d(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r

class RelevancePropagationBatchNorm1d(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.tensor, r: torch.tensor) -> torch.tensor:
        return r
"""Class for layer-wise relevance propagation.

Layer-wise relevance propagation for VGG-like networks from PyTorch's Model Zoo.
Implementation can be adapted to work with other architectures as well by adding the corresponding operations.

    Typical usage example:

        model = torchvision.models.vgg16(pretrained=True)
        lrp_model = LRPModel(model)
        r = lrp_model.forward(x)

"""
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from .utils import layers_lookup


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!
        # Parse network
        self.layers = self._get_layer_operations()

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.

        Returns:
            LRP-model as module list.

        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        layers = deepcopy(self.layers)
        reverse_layers = dict()
        lookup_table = layers_lookup()

        # Run backwards through layers
        reverse_layers['fcn'] = lookup_table[layers['fcn'].__class__](layer=layers['fcn'])
        reverse_layers['avgpool'] = lookup_table[layers['avgpool'].__class__](layer=layers['avgpool'])
        reverse_layers['data_bn'] = lookup_table[layers['data_bn'].__class__](layer=layers['data_bn'])
        for i in range(4):
            reverse_layers['st_gcn'+str(i)] = dict()
            reverse_layers['st_gcn'+str(i)]['gcn'] = torch.nn.ModuleList()
            for gcn_item in layers['st_gcn'+str(i)]['gcn']:
                reverse_layers['st_gcn'+str(i)]['gcn'].append(
                    lookup_table[gcn_item.__class__](layer=gcn_item)
                )
            reverse_layers['st_gcn'+str(i)]['tcn'] = torch.nn.ModuleList()
            for tcn_item in layers['st_gcn'+str(i)]['tcn']:
                reverse_layers['st_gcn'+str(i)]['tcn'].append(
                    lookup_table[tcn_item.__class__](layer=tcn_item)
                )
            reverse_layers['st_gcn'+str(i)]['relu'] = \
                lookup_table[layers['st_gcn'+str(i)]['relu'].__class__](layer=layers['st_gcn'+str(i)]['relu'])
            
            
            if isinstance(layers['st_gcn'+str(i)]['res'], nn.Sequential):
                reverse_layers['st_gcn'+str(i)]['res'] = torch.nn.ModuleList()
                for res_item in layers['st_gcn'+str(i)]['res']:
                    reverse_layers['st_gcn'+str(i)]['res'].append(
                        lookup_table[res_item.__class__](layer=res_item)
                    )
            else:
                reverse_layers['st_gcn'+str(i)]['res'] = None
        # for key, val in reverse_layers.items():
        #     print(key, ':', val)
        return reverse_layers

    def _get_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.

        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.

        Returns:
            Layers of original model stored in module list.

        """
        # edge_importances = []
        # for ei in self.model.edge_importance:
        #     edge_importances.append(ei)

        st_gcns = dict()
        st_gcns['data_bn'] = self.model.data_bn
        for i, st_gcn in enumerate(self.model.st_gcn_networks):
            st_gcns['st_gcn'+str(i)] = dict()
            
            st_gcns['st_gcn'+str(i)]['gcn'] = torch.nn.ModuleList()
            st_gcns['st_gcn'+str(i)]['gcn'].append(st_gcn.gcn.conv)
            # if i == 0:
            #     print('building weight', st_gcns['st_gcn'+str(i)]['gcn'][0].weight[:20])
            adj = self.model.A * self.model.edge_importance[i]
            k, v, w = adj.size()
            adj = adj.permute(2, 0, 1).contiguous().view(w, k*v)
            adj_as_w = nn.Linear(k * v, w, bias=False)
            adj_as_w.weight = torch.nn.Parameter(adj)
            st_gcns['st_gcn'+str(i)]['gcn'].append(adj_as_w)

            st_gcns['st_gcn'+str(i)]['tcn'] = st_gcn.tcn
            st_gcns['st_gcn'+str(i)]['relu'] = st_gcn.relu
            st_gcns['st_gcn'+str(i)]['res'] = st_gcn.residual

        st_gcns['avgpool'] = torch.nn.AdaptiveAvgPool2d((1, 1))
        st_gcns['fcn'] = self.model.fcn

        return st_gcns

    def forward(self, x: torch.tensor, call_layer:str) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.

        Args:
            x: Input tensor representing an image / images (N, C, H, W).

        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).

        """
        activations = dict()
        kernel_size = 3
        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations['input'] = torch.ones_like(x)
            # for layer in self.layers:
            #     x = layer.forward(x)
            #     activations.append(x)
            N, C, T, V, M = x.size()
            x = x.permute(0, 4, 3, 1, 2).contiguous()
            x = x.view(N * M, V * C, T)
            x = self.layers['data_bn'](x)
            x = x.view(N, M, V, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(N * M, C, T, V)
            activations['data_bn'] = x # bn activation
            
            for i in range(4):
                activations['st_gcn'+str(i)] = dict()

                # residual activation
                activations['st_gcn'+str(i)]['res'] = list()
                res = x
                if isinstance(self.layers['st_gcn'+str(i)]['res'], nn.Sequential):
                    for res_item in self.layers['st_gcn'+str(i)]['res']:
                        res = res_item(res)
                        activations['st_gcn'+str(i)]['res'].append(res)
                else:
                    res = self.layers['st_gcn'+str(i)]['res'](res)
                    activations['st_gcn'+str(i)]['res'].append(res)

                # gcn activation
                activations['st_gcn'+str(i)]['gcn'] = list()
                x = self.layers['st_gcn'+str(i)]['gcn'][0](x)
                activations['st_gcn'+str(i)]['gcn'].append(x)
                
                n, kc, t, v = x.size()
                x = x.view(n, kernel_size, kc//kernel_size, t, v)
                x = x.permute(0,2,3,1,4).contiguous().view(-1, kernel_size*v)
                x = self.layers['st_gcn'+str(i)]['gcn'][1](x)
                x = x.view(n, kc//kernel_size, t, v)
                activations['st_gcn'+str(i)]['gcn'].append(x)

                # tcn activation
                activations['st_gcn'+str(i)]['tcn'] = list()
                for tcn_item in self.layers['st_gcn'+str(i)]['tcn']:
                    x = tcn_item(x)
                    activations['st_gcn'+str(i)]['tcn'].append(x)

                # relu activation
                x = self.layers['st_gcn'+str(i)]['relu'](x + activations['st_gcn'+str(i)]['res'][-1])
                activations['st_gcn'+str(i)]['relu'] = x
            
            
            bs, c, t, v = x.size()
            x = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(N, c, t*v, M)
            x = self.layers['avgpool'](x)
            activations['avgpool'] = x
            x = self.layers['fcn'](x)
            activations['fcn'] = x

        r = torch.softmax(activations['fcn'], dim=1)  # Unsupervised
        # print(activations['fcn'][:,:20,:,:])
        mask = (r == r.max(dim=1, keepdim=True)[0])
        r = r * mask

        # Perform relevance propagation
        relevance = dict()
        
        r = self.lrp_layers['fcn'].forward(activations['avgpool'].requires_grad_(True), r)
        relevance['fcn'] = r
        act = activations['st_gcn'+str(3)]['relu'].view(N, M, c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        act = act.view(N, c, t*v, M)
        
        r = self.lrp_layers['avgpool'].forward(act.requires_grad_(True), r)
        relevance['avgpool'] = r
        r = r.permute(0, 3, 1, 2).contiguous()
        r = r.view(N * M, c, t, v)

        for i in range(4):
            relevance['st_gcn'+str(i)] = dict()

        for i in range(3, -1, -1):
            # relu 的输入为tcn[-1]和res[-1]
            tcn_act = activations['st_gcn'+str(i)]['tcn'][-1]
            res_act = activations['st_gcn'+str(i)]['res'][-1]  if i else torch.zeros_like(tcn_act)
            sum_act_tcn_res = tcn_act + res_act
            r = self.lrp_layers['st_gcn'+str(i)]['relu'].forward(
                sum_act_tcn_res.requires_grad_(True), r)
            relevance['st_gcn'+str(i)]['relu'] = r

            tcn_r = r * tcn_act / (sum_act_tcn_res + 1e-7)
            res_r = r * res_act / (sum_act_tcn_res + 1e-7)

            # tcn relevance
            for j in range(len(self.lrp_layers['st_gcn'+str(i)]['tcn'])-1, 0, -1):
                tcn_r = self.lrp_layers['st_gcn'+str(i)]['tcn'][j].forward(
                    activations['st_gcn'+str(i)]['tcn'][j - 1].data.requires_grad_(True),
                    tcn_r
                )
            tcn_r = self.lrp_layers['st_gcn'+str(i)]['tcn'][0].forward(
                activations['st_gcn'+str(i)]['gcn'][-1].data.requires_grad_(True),
                tcn_r
            )
            relevance['st_gcn'+str(i)]['tcn'] = tcn_r # 舍弃tcn relevance中间计算结果

            # gcn relevance
            gcn_r = tcn_r
            act = activations['st_gcn'+str(i)]['gcn'][0]
            n, kc, t, v = act.size()
            act = act.view(n, kernel_size, kc//kernel_size, t, v)
            act = act.permute(0, 2, 3, 1, 4).contiguous().view(-1, kernel_size*v)
            gcn_r = gcn_r.view(n * kc//kernel_size * t, v)
            gcn_r = self.lrp_layers['st_gcn'+str(i)]['gcn'][1].forward(act.data.requires_grad_(True), gcn_r)
            
            gcn_r = gcn_r.view(n, kc//kernel_size, t, kernel_size, v)
            gcn_r = gcn_r.permute(0, 3, 1, 2, 4).contiguous()
            gcn_r = gcn_r.view(n, kc, t, v)
            gcn_temp = gcn_r
            gcn_r = self.lrp_layers['st_gcn'+str(i)]['gcn'][0].forward(
                activations['st_gcn'+str(i-1)]['relu'].requires_grad_(True) if i else activations['data_bn'].requires_grad_(True), 
                gcn_r
            )
            relevance['st_gcn'+str(i)]['gcn'] = gcn_r

            # res relevance
            if self.lrp_layers['st_gcn'+str(i)]['res'] is not None:
                
                res_r = self.lrp_layers['st_gcn'+str(i)]['res'][1].forward(
                    activations['st_gcn'+str(i)]['res'][0],
                    res_r
                )
                res_r = self.lrp_layers['st_gcn'+str(i)]['res'][0].forward(
                    activations['st_gcn'+str(i-1)]['relu'], # if i else activations['data_bn'],
                    res_r
                )
            relevance['st_gcn'+str(i)]['res'] = res_r if i else torch.zeros_like(relevance['st_gcn'+str(i)]['gcn'])
            r = relevance['st_gcn'+str(i)]['res'] + relevance['st_gcn'+str(i)]['gcn']
            
        r = r.view(N, M, C, T, V).permute(0, 1, 4, 2, 3).contiguous()
        r = r.view(N * M, V * C, T)
        r = self.lrp_layers['data_bn'].forward(activations['input'], r)
        r = r.view(N, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous()
        relevance['data_bn'] = r
        
        visual = relevance['st_gcn'+call_layer]['gcn']
        n, c, t, v = visual.size()
        visual = visual.view(N, M, c, t, v).permute(0, 2, 3, 4, 1).contiguous()

        visual = F.relu(visual)
        visual = F.interpolate(visual, size=(T, V, M), mode='trilinear', align_corners=True)
        return visual.sum(dim=1, keepdim=True).detach()

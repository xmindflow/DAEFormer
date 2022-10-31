import numpy as np
import torch
from pytorch_model_summary import summary

from experiment_depth.depth_3_block_6 import ChannelEffFormer
from missformer_nets.missformer import MISSFormer
from networks.ChannelFuseFormerAdd1 import ChannelFuseFormerAdd1
from networks.ChannelFuseFormerAdd2 import ChannelFuseFormerAdd2
from networks.ChannelFuseFormerCat import ChannelFuseFormerCat
from networks.SpectralSpatialFormer import encoder


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


net = ChannelEffFormer(num_classes=9).cuda(0)
net.load_state_dict(torch.load("results/depth_3_block_6/transfilm_epoch_399.pth"))
print(net)
print(summary(ChannelEffFormer(), torch.zeros((3, 1, 224, 224)), show_input=True, batch_size=24))

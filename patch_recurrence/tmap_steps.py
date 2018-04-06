"""Transmission map estimation functions."""
import numpy as np

import sys

import logging

import torch

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor    # uncomment this line for GPU


def gradient(x):
    """Returns x and y components of grad(x)."""
    diff1 = torch.Tensor([-1, 0, 1]).type(dtype)
    diff1 = Variable(diff1.view(1, 1, 1, 3), requires_grad=False)
    diff2 = torch.Tensor([1, 0, -1]).type(dtype)
    diff2 = Variable(diff2.view(1, 1, 3, 1), requires_grad=False)
    conv1 = F.conv2d(x, diff1, padding=(0, 1))
    conv2 = F.conv2d(x, diff2, padding=(1, 0))
    return conv1, conv2


def w_val(l_img):
    l_img = torch.mean(l_img, 2)
    height, width = l_img.size(0), l_img.size(1)
    l_img = Variable(l_img.view(1, 1, height, width), requires_grad=False)
    conv1, conv2 = gradient(l_img)
    raised = (0.1 - torch.sqrt(torch.mul(conv1, conv1) + torch.mul(conv2, conv2))) * 48
    result = torch.sigmoid(raised)
    result.view(height * width)
    return result


class Net(nn.Module):
    def __init__(self, img, patches, airlight, tlb, constants):
        super(Net, self).__init__()
        self.img = img
        self.patches = patches
        self.airlight = airlight
        self.constants = constants
        self.tmap = Variable(tlb, requires_grad=True)

    def get_norm(self, x):
        height, width = x.size(0), x.size(1)
        log = torch.log(torch.clamp(x, min=0.0000001, max=1))
        log = log.view(1, 1, height, width)
        conv1, conv2 = gradient(log)
        l2_norm = torch.mul(conv1, conv1) + torch.mul(conv2, conv2)
        l2_norm.view(height * width)
        return l2_norm

    def forward(self, l_img):
        """Equation (26)"""
        tmap = self.tmap
        l2_norm = self.get_norm(tmap)
        sig = w_val(l_img)
        s = l2_norm * sig
        return torch.sum(s)


def estimate_tmap(img, patches, airlight, constants):
    """Estimates t-map and returns dehazed output image after 20 iterations
    """
    patch_size = constants.PATCH_SIZE
    h, w = img.shape[0], img.shape[1]

    # Intialize tmap as tlb
    tlb = np.empty([len(patches)])
    for index, patch in enumerate(patches):
        raw = np.reshape(patch.raw_patch, [-1, 3])
        tlb_patch = 1 - raw / airlight
        tlb[index] = max(tlb_patch[patch_size ** 2 // 2])
    tlb = np.reshape(tlb, [h - patch_size, w - patch_size, 1])

    img = np.reshape(img[0:h - patch_size, 0:w - patch_size], [h - patch_size, w - patch_size, 3])
    l_img = (img - airlight) / tlb + airlight

    # Define the Network
    net = Net(img, patches, airlight, torch.from_numpy(tlb).type(dtype), constants)
    # net.cuda()    # uncomment this line for GPU

    # Actual Optimization
    optimizer = torch.optim.SGD([net.tmap], lr=0.001)
    for i in range(100):
        optimizer.zero_grad()
        loss = net(torch.from_numpy(l_img).type(dtype))
        logger.info("Loss is %f", loss)
        loss.backward()
        optimizer.step()
        tmap = net.tmap.data
        # tmap = tmap.cpu().numpy()    # uncomment this line for GPU and comment next line
        tmap = tmap.numpy()
        tmap = np.reshape(tmap, [h - patch_size, w - patch_size, 1])
        l_img = (img - airlight) / tmap + airlight

    l_img = np.reshape(l_img, [h - patch_size, w - patch_size, 3])
    return l_img

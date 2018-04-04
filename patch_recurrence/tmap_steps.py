"""Transmission map estimation functions."""
import numpy as np

import math

from decimal import Decimal

import sys

import logging

import torch

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

import tools

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU

class Net(nn.Module):
    def __init__(self, img, patches, airlight, constants):
        super(Net, self).__init__()
        self.img = img
        self.patches = patches
        self.airlight = airlight
        self.constants = constants

    def sigmoid(self, y, length):
        sig = torch.Tensor(length).type(dtype).zero_()
        for i in range(length):
            l2_norm = math.sqrt(sum(map(lambda x: x * x, y[i])))
            res = 1 / (1 + Decimal(48 * (l2_norm - 0.1)).exp())
            sig[i] = float(res)
        return sig

    def get_norm(self, x):
        height, width = x.size(0), x.size(1)

        log = torch.log(x)
        log = log.view(1, 1, height, width)
        diff1 = torch.Tensor([-1, 0, 1]).type(dtype)
        diff1 = Variable(diff1.view(1, 1, 1, 3), requires_grad=False)
        diff2 = torch.Tensor([1, 0, -1]).type(dtype)
        diff2 = Variable(diff2.view(1, 1, 3, 1), requires_grad=False)

        # conv1 and conv2 should contain the x and y components resp, of grad(log)
        conv1 = F.conv2d(log, diff1, padding=(0, 1))
        conv2 = F.conv2d(log, diff2, padding=(1, 0))

        l2_norm = torch.mul(conv1, conv1) + torch.mul(conv2, conv2)
        l2_norm.view(height * width)
        return l2_norm

    def loss_fun(self, w, sig):
        """Equation (26)"""
        l2_norm = self.get_norm(w)
        s = l2_norm * sig
        ret = torch.sum(s)
        return ret

    def minimization(self, sig, tlb, rate):
        """Returns new t-map
        Initialize tmap with tlb
        sig is a constant for the optimization step
        """
        t_height, t_width = len(tlb), len(tlb[0])
        tlb = torch.Tensor(np.reshape(tlb, [t_height, t_width])).type(dtype)
        for i in range(t_height):
            for j in range(t_width):
                if tlb[i][j] <= 0:
                    tlb[i][j] = 10 ** -7

        weight = Variable(tlb, requires_grad=True)
        sig = Variable(sig, requires_grad=False)

        optimizer = torch.optim.SGD([weight], lr=rate)
        optimizer.zero_grad()
        loss_val = loss = self.loss_fun(weight, sig)
        logger.info("Loss is %f. Rate is %f", loss, rate)
        loss.backward()
        optimizer.step()
        t = weight.data
        return t, loss_val

    def forward(self):
        patch_size = self.constants.PATCH_SIZE
        h, w = self.img.shape[0], self.img.shape[1]

        # Initializing lower bounded transmission map tlb - equn(15)
        tlb = np.empty([len(self.patches)])
        for index, patch in enumerate(self.patches):
            raw = np.reshape(patch.raw_patch, [-1, 3])
            tlb_patch = 1 - raw / self.airlight
            tlb[index] = max(tlb_patch[patch_size ** 2 // 2])
        tlb = np.reshape(tlb, [h - patch_size, w - patch_size, 1])
        img = np.reshape(self.img[0:h - patch_size, 0:w - patch_size], [h - patch_size, w - patch_size, 3])
        l_img = (img - self.airlight) / tlb + self.airlight

        # Initial Sigmoid calculation
        l_img = np.reshape(l_img, [3, -1])
        grad = [np.gradient(l_img[i]) for i in range(3)]
        grad = np.reshape(grad, [-1, 3])
        l_img = np.reshape(l_img, [-1, 3])
        sig = self.sigmoid(torch.from_numpy(grad), len(l_img))
        sig = sig.view(h - patch_size, w - patch_size)

        # Run through 100 iterations
        t_prev = tlb
        new_grad = np.empty([3, l_img.shape[0] * l_img.shape[1]])

        rate = 0.001
        loss_list = []
        for i in range(100):
            t_curr, curr_loss = self.minimization(sig, t_prev, rate)
            loss_list.append(curr_loss.data[0])
            t_curr = t_curr.cpu().numpy()
            t_curr = np.reshape(t_curr, [h - patch_size, w - patch_size, 1])
            l_img = (img - self.airlight) / t_curr + self.airlight
            t_prev = t_curr

            # Recalculation of sigmoid
            l_img = np.reshape(l_img, [3, -1])
            new_grad = [np.gradient(l_img[i]) for i in range(3)]
            grad = np.reshape(new_grad, [-1, 3])
            l_img = np.reshape(l_img, [-1, 3])
            sig = self.sigmoid(grad, len(l_img))
            sig = sig.view(h - patch_size, w - patch_size)

        # tools.show_loss(loss_list, 100, 'SGD Optimization Algo')
        # tools.show_tmap([t_prev])
        l_img = np.reshape(l_img, [h - patch_size, w - patch_size, 3])
        return l_img


def estimate_tmap(img, patches, airlight, constants):
    """Estimates t-map and returns dehazed output image after 20 iterations
    """
    net = Net(img, patches, airlight, constants)
    net.cuda()
    res = net()
    return res

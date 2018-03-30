"""Transmission map estimation functions."""
import numpy as np

import math

from decimal import Decimal

import torch

from torch.nn.parameter import Parameter

from torch.autograd import Variable

import torch.nn.functional as F

import tools


def sigmoid(y):
    """Returns sigmoid value of pixel x
    """
    l2_norm = math.sqrt(sum(map(lambda x: x * x, y)))
    res = 1 / (1 + Decimal(48 * (l2_norm - 0.1)).exp())
    return res


def get_norm(x):
    height, width = x.size(0), x.size(1)
    log = torch.log(x)

    log = log.view(1, 1, height, width)
    diff1 = torch.Tensor([-1, 0, 1])
    diff1 = Variable(diff1.view(1, 1, 1, 3), requires_grad=True)
    diff2 = torch.Tensor([1, 0, -1])
    diff2 = Variable(diff2.view(1, 1, 3, 1), requires_grad=True)

    # conv1 and conv2 should contain the x and y components resp, of grad(log)
    conv1 = F.conv2d(log, diff1, padding=(0, 1))
    conv2 = F.conv2d(log, diff2, padding=(1, 0))

    l2_norm = torch.mul(conv1, conv1) + torch.mul(conv2, conv2)
    l2_norm.data.resize_(height * width)
    return l2_norm


def loss_fun(x, sig):
    """Equation (26)"""
    l2_norm = get_norm(x)
    sig2 = np.asarray(sig, dtype=np.float32)
    sig1 = torch.from_numpy(sig2)
    t = torch.mul(l2_norm.data, sig1)
    ret = Variable(torch.FloatTensor([torch.sum(t)]), requires_grad=True)
    return ret


def minimization(sig, tlb):
    """Returns new t-map
    """
    t_height, t_width = len(tlb), len(tlb[0])
    tlb = torch.FloatTensor(np.reshape(tlb, [t_height, t_width]))
    for i in range(t_height):
        for j in range(t_width):
            if tlb[i][j] <= 0:
                tlb[i][j] = 10 ** -7

    weight = Parameter(torch.Tensor(t_height, t_width), requires_grad=True)
    weight.data = tlb

    optimizer = torch.optim.SGD([weight], lr=100)
    loss = loss_fun(weight, sig)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t = weight.data
    return t


def estimate_tmap(img, patches, airlight, constants):
    """Estimates t-map and returns dehazed output image after 20 iterations
    """
    patch_size = constants.PATCH_SIZE
    h, w = img.shape[0], img.shape[1]

    # Initializing lower bounded transmission map tlb
    tlb = np.empty([len(patches)])
    for index, patch in enumerate(patches):
        raw = np.reshape(patch.raw_patch, [-1, 3])
        tlb_patch = 1 - raw / airlight
        tlb[index] = max(tlb_patch[patch_size ** 2 // 2])
    tlb = np.reshape(tlb, [h - patch_size, w - patch_size, 1])
    img = np.reshape(img[0:h - patch_size, 0:w - patch_size], [h - patch_size, w - patch_size, 3])
    l_img = (img - airlight) / tlb + airlight

    # Sigmoid calculation
    grad = np.empty([3, l_img.shape[0] * l_img.shape[1]])
    l_img = np.reshape(l_img, [3, -1])
    grad[0] = np.gradient(l_img[0])
    grad[1] = np.gradient(l_img[1])
    grad[2] = np.gradient(l_img[2])
    grad = np.reshape(grad, [-1, 3])
    l_img = np.reshape(l_img, [-1, 3])
    sig = torch.Tensor([float(sigmoid(grad[i])) for i in range(len(l_img))])

    # Run through 10 iterations
    t_prev = tlb
    for i in range(10):
        t_curr = minimization(sig, t_prev)
        t_curr = t_curr.numpy()
        t_curr = np.reshape(t_curr, [h - patch_size, w - patch_size, 1])
        l_img = (img - airlight) / t_curr + airlight
        t_prev = t_curr
        tools.show_img([img, l_img])

    return l_img

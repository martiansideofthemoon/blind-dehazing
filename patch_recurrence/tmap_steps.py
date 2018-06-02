"""Transmission map estimation functions."""
import numpy as np

import sys

import logging

import steps

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


class Net(nn.Module):
    """Optimization class."""

    def __init__(self, img, pairs, raw_pairs, airlight, tlb):
        """Initialize."""
        super(Net, self).__init__()
        self.img = img
        self.pairs = pairs
        self.raw_pairs = raw_pairs
        self.airlight = airlight
        self.tmap = Variable(tlb, requires_grad=True)

    def w_val(self, l_img):
        """Decreasing sigmoid function."""
        l_img = torch.mean(l_img, 2)
        height, width = l_img.size(0), l_img.size(1)
        l_img = Variable(l_img.view(1, 1, height, width), requires_grad=False)
        conv1, conv2 = self.spatial_gradient(l_img)
        raised = (0.1 - torch.sqrt(torch.mul(conv1, conv1) + torch.mul(conv2, conv2))) * 48
        result = torch.sigmoid(raised)
        result.view(height * width)
        return result

    def spatial_gradient(self, x):
        """Return x and y components of grad(x)."""
        diff1 = torch.Tensor([-1, 0, 1]).type(dtype)
        diff1 = Variable(diff1.view(1, 1, 1, 3), requires_grad=False)
        diff2 = torch.Tensor([1, 0, -1]).type(dtype)
        diff2 = Variable(diff2.view(1, 1, 3, 1), requires_grad=False)
        conv1 = F.conv2d(x, diff1, padding=(0, 1))
        conv2 = F.conv2d(x, diff2, padding=(1, 0))
        return conv1, conv2

    def gradient(self, x):
        """Return l2-norm squared for grad(log(t))."""
        height, width = x.size(0), x.size(1)
        log = torch.log(torch.clamp(x, min=0.0000001, max=1))
        log = log.view(1, 1, height, width)
        conv1, conv2 = self.spatial_gradient(log)
        l2_norm = torch.mul(conv1, conv1) + torch.mul(conv2, conv2)
        l2_norm.view(height * width)
        return l2_norm

    def get_patch_recurr(self):
        """Equation (24)."""
        t = self.tmap
        ret = Variable(torch.Tensor(len(self.pairs)).type(dtype), requires_grad=False)
        for i, pair in enumerate(self.pairs):
            first, second = pair.first, pair.second
            air_free1 = Variable(torch.from_numpy(first.raw_patch - self.airlight).type(dtype), requires_grad=False)
            air_free2 = Variable(torch.from_numpy(second.raw_patch - self.airlight).type(dtype), requires_grad=False)
            location1, location2 = first.location, second.location
            x = t[location2[0]][location2[1]] * air_free1 - t[location1[0]][location1[1]] * air_free2
            ret[i] = torch.norm(x, 2) ** 2
        return torch.sum(ret)

    def get_raw(self, pairs):
        """Equation (25)."""
        t = self.tmap
        h, w = len(t), len(t[0])
        t = t.view(h * w)
        first = [pair.first.location for pair in pairs]
        second = [pair.second.location for pair in pairs]
        f_index = [f[0] * w + f[1] for f in first]
        s_index = [s[0] * w + s[1] for s in second]
        f_tval = torch.gather(t, 0, Variable(torch.LongTensor(f_index), requires_grad=False))
        s_tval = torch.gather(t, 0, Variable(torch.LongTensor(s_index), requires_grad=False))
        tval = s_tval - f_tval
        square = torch.mul(tval, tval)
        return torch.sum(square)

    def get_smooth(self, l_img):
        """Equation (26)."""
        tmap = self.tmap
        l2_norm = self.gradient(tmap)
        sig = self.w_val(l_img)
        s = l2_norm * sig
        return torch.sum(s)

    def forward(self, l_img):
        """Equation (23)."""
        smooth = self.get_smooth(l_img)
        patch_recurr = self.get_patch_recurr()
        raw = self.get_raw(self.raw_pairs)
        print smooth.data[0], patch_recurr.data[0], raw.data[0]
        patch_recurrence = torch.add(patch_recurr, raw)
        return torch.add(smooth * 0.5, patch_recurrence)


def estimate_tmap(img, patches, pairs, airlight, constants):
    """Estimate t-map and returns dehazed output image after 20 iterations."""
    patch_size = constants.PATCH_SIZE
    h, w = img.shape[0], img.shape[1]

    # Intialize tmap as tlb
    tlb = np.empty([len(patches[0])])
    for index, patch in enumerate(patches[0]):
        raw = np.reshape(patch.raw_patch, [-1, 3])
        tlb_patch = 1 - raw / airlight
        tlb[index] = max(tlb_patch[patch_size ** 2 // 2])
    tlb = np.reshape(tlb, [h - patch_size, w - patch_size, 1])

    img = np.reshape(img[0:h - patch_size, 0:w - patch_size], [h - patch_size, w - patch_size, 3])
    l_img = (img - airlight) / tlb + airlight
    # return l_img

    logger.info("Generating raw pairs of patches ...")
    raw_pairs = steps.generate_pairs_raw(patches, constants)
    print("\nNumber of pairs generated using generate_pairs")
    print(len(raw_pairs))
    logger.info("Removing duplicates ...")
    raw_pairs = steps.remove_duplicates(raw_pairs)
    print("\nNumber of pairs after duplicate removal")
    print(len(raw_pairs))
    logger.info("Forming pair ...")
    raw_pairs = steps.filter_pairs(patches, raw_pairs, constants, all_pairs=True)
    # Use `patches` and find all low-SSD pairs and store as raw_pairs instead of above steps

    # Define the Network
    net = Net(img, pairs, raw_pairs, airlight, torch.from_numpy(tlb).type(dtype))
    # net.cuda()    # uncomment this line for GPU

    # Actual Optimization
    optimizer = torch.optim.SGD([net.tmap], lr=0.00001)
    for i in range(1000):
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

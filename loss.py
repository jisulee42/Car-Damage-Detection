import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def miou():
    # # forward pass
    # img = data['img'].to(device)
    # label = data['label'].to(device)
    # output = net(img)

    # label_temp = label.squeeze(1)
    # output_temp = output.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    # intersection = (output_temp & label_temp).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    # union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0    
    # iou_score = np.sum(intersection) / np.sum(union)
    
    # iou_scores.append(iou_score)
    # # print("TEST: BATCH %04d \ %04d | IoU %.4f" % (batch, num_batch_test, iou_score))
    # print("TEST: BATCH %04d \ %04d" % (batch, num_batch_test))
    pass

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    loss_1.data = loss_1.data.cpu() # !@#
    ind_1_sorted = np.argsort(loss_1.data).cuda() # [1.5, 0.2, 4.2, 2.5] -> [1, 0, 3, 2]
    loss_1_sorted = loss_1[ind_1_sorted] # [1.5, 0.2, 4.2, 2.5] -> [0.2, 1.5, 2.5, 4.2]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    loss_2.data = loss_2.data.cpu() # !@#
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_sorted = ind_1_sorted.cpu() # !@#
    ind_2_sorted = ind_2_sorted.cpu() # !@#

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2



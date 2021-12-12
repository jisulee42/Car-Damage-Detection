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
def co_teaching_loss(logits1, logits2, label, rt):

    fn_loss = nn.BCEWithLogitsLoss()

    # 배치별 로스를 텐서로 변환(배치 4면 길이 4인 텐서 생성)
    model1_loss, model2_loss = [], []
    for i, j, l  in zip(logits1, logits2, label):
        model1_loss.append(fn_loss(i, l))
        model2_loss.append(fn_loss(j, l))

    model1_loss = torch.tensor(model1_loss, requires_grad=True).cuda()
    model2_loss = torch.tensor(model2_loss, requires_grad=True).cuda()

    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0

    model1_loss = (model1_loss_filter * model1_loss).sum()
    
    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss
    

def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    fn_loss = nn.BCEWithLogitsLoss(reduce=False).cuda()
    #fn_loss = nn.BCEWithLogitsLoss().cuda()
    # fn_loss = nn.BCEWithLogitsLoss(y_1, t, reduce=False)
    # nn.MSELoss(y_1, t, reduce=False)
    loss_1 = fn_loss(y_1, t)
    print(loss_1)
    loss_1.data = loss_1.data.cpu() # !@#
    ind_1_sorted = np.argsort(loss_1.data).cuda() # [1.5, 0.2, 4.2, 2.5] -> [1, 0, 3, 2]
    loss_1_sorted = loss_1[ind_1_sorted] # [1.5, 0.2, 4.2, 2.5] -> [0.2, 1.5, 2.5, 4.2]

    loss_2 = fn_loss(y_2, t)
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
    loss_1_update = fn_loss(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = fn_loss(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
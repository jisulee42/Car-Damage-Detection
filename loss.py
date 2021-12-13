import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)
        
        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        ## Return average loss over classes and batch
        return 1-loss.mean()

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
def co_teaching_loss(loss, device, output1, output2, label, rt):

    # Defining Loss Function
    if loss == 'focalloss':
        criterion = FocalLoss(gamma=3/4).to(device)
    elif loss == 'iouloss':
        criterion = mIoULoss(n_classes=2).to(device)
    elif loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    elif loss == 'bcelogitsloss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        print('Loss function not found!')

    # 배치별 로스를 텐서로 변환(배치 4면 길이 4인 텐서 생성)
    model1_loss, model2_loss = [], []
    for i, j, lbl  in zip(output1, output2, label):
        model1_loss.append(criterion(i, lbl))
        model2_loss.append(criterion(j, lbl))

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
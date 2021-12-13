# Importing Libraries
import os
import torch


# Saving Network
def save(ckpt_dir, net1, net2, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net1.state_dict(), 'optim': optim.state_dict()},
               "%s/model1_epoch%d.pth" % (ckpt_dir, epoch))

    torch.save({'net2': net1.state_dict(), 'optim': optim.state_dict()},
               "%s/model2_epoch%d.pth" % (ckpt_dir, epoch))
    


# Loading Network
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
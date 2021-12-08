# Importing Libraries
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import torch.nn.functional as F
#import wandb
#wandb.init(project="U-Net", entity="kim1lee3")

#from model._unet import UNet
from model import UNet
from dataset import *
from util import *
from loss import loss_coteaching

# Parsing Inputs
parser = argparse.ArgumentParser(description="Train the Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="../datasets_gray_npy/dent", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint/dent", type=str, dest="ckpt_dir")

parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')

args = parser.parse_args()


# Setting Variables
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir

train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb.config = {
#   "learning_rate": args.lr,
#   "epochs": args.num_epoch,
#   "batch_size": args.batch_size
# }

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

# define drop rate schedule
args.exponent = 1 # !@#
rate_schedule = np.ones(num_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)



# Network
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset_train = Dataset(img_dir=os.path.join(data_dir, 'train', 'images'), \
                              label_dir=os.path.join(data_dir, 'train', 'masks'), \
                              transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(img_dir=os.path.join(data_dir, 'valid', 'images'), \
                            label_dir=os.path.join(data_dir, 'valid', 'masks'), \
                            transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# 전부 True로 채우기
noise_or_not = np.full(num_data_train, True)

# Creating Network
net = UNet().to(device)
net2 = UNet().to(device)


# Defining Loss Function
fn_loss = nn.BCEWithLogitsLoss().to(device)


# Defining Optimizer
optim1 = torch.optim.Adam(net.parameters(), lr=lr)
optim2 = torch.optim.Adam(net2.parameters(), lr=lr)


# Train
st_epoch = 0

if train_continue == "on":
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
# ---
pure_ratio_list=[]
pure_ratio_1_list=[]
pure_ratio_2_list=[]

train_total=0
train_correct=0 
train_total2=0
train_correct2=0 

def train():
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_train = []

        for batch, (data, indexes) in enumerate(loader_train, 1):
            ind = indexes.cpu().numpy().transpose()
            # forward pass
            img = data['img'].to(device)
            label = data['label'].to(device)
            # Forward
            logits1 = net(img)
            prec1, _ = accuracy(logits1, label, topk=(1,5))
            train_total += 1
            train_correct += prec1

            logits2 = net2(img)
            prec2, _ = accuracy(logits2, label, topk=(1,5))
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, label, rate_schedule[epoch], ind, noise_or_not)
            pure_ratio_1_list.append(100*pure_ratio_1)
            pure_ratio_2_list.append(100*pure_ratio_2)
            # backward Backward Optimize
            
            optim1.zero_grad()
            loss_1.backward()
            optim1.step()
            optim2.zero_grad()
            loss_2.backward()
            optim2.step()

            # calculate loss function
            loss_train += [loss.item()]
            if batch%10 == 0:
                print("Train: EPOCH %04d / %04d | BATCH %04d \ %04d | LOSS %.4f"
                    % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_train)))

        # wandb.log({"train loss": np.mean(loss_train)})

        with torch.no_grad():
            net.eval()
            loss_valid = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                img = data['img'].to(device)
                label = data['label'].to(device)
                output = net(img)

                # calculate loss function
                loss = fn_loss(output, label)
                loss_valid += [loss.item()]
                if batch%10 == 0:
                    print("Validation: EPOCH %04d / %04d | BATCH %04d \ %04d | LOSS %.4f"
                        % (epoch, num_epoch, batch, num_batch_val, np.mean(loss_valid)))

        # wandb.log({"valid loss": np.mean(loss_valid)})

        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

if __name__ == "__main__":
    train()
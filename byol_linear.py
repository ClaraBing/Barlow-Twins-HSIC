import argparse

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# from thop import profile, clever_format
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models
from tqdm import tqdm
from byol import BYOL
import utils
from utils import str2bool

import torchvision

import pdb

EARLY_STOP_EPS = 0.1

try:
  import wandb
  USE_WANDB = True
except Exception as e:
  print('Exception:', e)
  print('Not using wandb. \n\n')
  USE_WANDB = False

if torch.cuda.is_available():
  torch.backends.cudnn.benchmark = True
  device = 'cuda'
else:
  device = 'cpu'


class Net(nn.Module):
    def __init__(self, encoder, num_class, fname_linear_cls=''):
        super(Net, self).__init__()

        # encoder
        self.f = encoder
        # classifier
        # self.fc = nn.Linear(2048, num_class, bias=True)
        # classifier
        if fname_linear_cls != '':
          linear_state_dict = torch.load(fname_linear_cls, map_location='cpu')
          # self.sanity_cls_weight, self.sanity_cls_bias = linear_state_dict['fc.weight'].clone().to(device), linear_state_dict['fc.bias'].clone().to(device)
          self.cls_weight, self.cls_bias = linear_state_dict['fc.weight'], linear_state_dict['fc.bias']
          self.cls_weight.requires_grad, self.cls_bias.requires_grad = False, False
          self.cls_weight, self.cls_bias = self.cls_weight.to(device), self.cls_bias.to(device)
          self.cls_weight = torch.nn.parameter.Parameter(self.cls_weight)
          self.cls_bias = torch.nn.parameter.Parameter(self.cls_bias)

          self.fc = nn.Linear(2048, 2048, bias=True)
        else:
          self.cls_weight, self.cls_bias = None, None
          self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x, return_projection = False)
        # pdb.set_trace()
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        bt_cnt = 0
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% model: {}'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100,
                                             model_path.split('/')[-1]))

            bt_cnt += 1
            if USE_WANDB and bt_cnt % 20 == 0:
              if is_train:
                wandb.log({
                  'loss':loss.item(),
                  'linear_total_correct_1': total_correct_1 / total_num * 100,
                  'linear_total_correct_5': total_correct_5 / total_num * 100,
                  })
              else:
                wandb.log({
                  'val_loss':loss.item(),
                  'val_linear_total_correct_1': total_correct_1 / total_num * 100,
                  'val_linear_total_correct_5': total_correct_5 / total_num * 100,
                  })
    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--model_path', type=str, default='results/Barlow_Twins/0.005_64_128_model.pth',
                        help='The base string of the pretrained model path')
    parser.add_argument('--fname-linear-cls', type=str, default='', help="File name of the ckpt that will provide a linear layer.")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of sweeps over the dataset to train')
    parser.add_argument('--image_size', default=32, type=int, help='Size of image')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--proj-head-type', default='2layer', choices=['none', 'linear', '2layer', 'linear_noBNReLU'],
                        help="Type of the projector.")
    parser.add_argument('--proj_hidden_dim', default=512, type=int, help='Feature dim for latent vector')
    parser.add_argument('--use_default_enc', type=str2bool, default=True, help="use default resnet 50 encoder")

    # optimization
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-6, type=float)

    # logging
    parser.add_argument('--project', default='nonContrastive')
    parser.add_argument('--wb-name', default='default', type=str,
                        help="Run name for wandb.")
    parser.add_argument('--wb-token', default='default', type=str)
    parser.add_argument('--wb-proj-type', default='byol', type=str,
                        help="For wandb filtering.")
    parser.add_argument('--overwrite', type=int, default=0,
                        help="Whether to overwrite an existing directory.")


    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    lr, wd = args.lr, args.wd

    if USE_WANDB:
      if args.wb_name != 'default':
        wandb.init(project=args.project, name=args.wb_name, config=args, entity='ssl-mld')
      else:
        wandb.init(project=args.project, config=args, entity='ssl-mld')

    dataset = args.dataset
    if dataset == 'cifar10':
        train_data = CIFAR10(root='data', train=True,\
            transform=utils.CifarPairTransform(train_transform = True, pair_transform=False), download=True)
        test_data = CIFAR10(root='data', train=False,\
            transform=utils.CifarPairTransform(train_transform = False, pair_transform=False), download=True)
    elif dataset == 'stl10':
        train_data =  torchvision.datasets.STL10(root='data', split="train", \
            transform=utils.StlPairTransform(train_transform = True, pair_transform=False), download=True)
        test_data =  torchvision.datasets.STL10(root='data', split="test", \
            transform=utils.StlPairTransform(train_transform = False, pair_transform=False), download=True)
    elif dataset == 'tiny_imagenet':
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                            utils.TinyImageNetPairTransform(train_transform=True, pair_transform=False))
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', \
                            utils.TinyImageNetPairTransform(train_transform = False, pair_transform=False))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.use_default_enc:
        encoder = models.resnet50()
        hidden_layer = 'avgpool'
    else:
        enc_layers = []
        for name, module in models.resnet50(pretrained=args.use_pretrained_enc).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10':
                #if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                if not isinstance(module, nn.MaxPool2d) and not isinstance(module, nn.Linear):
                    enc_layers.append(module)
                else:
                    print("skipping MaxPool2d... or Linear")
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    enc_layers.append(module)
        # encoder
        encoder = nn.Sequential(*enc_layers)
        hidden_layer = -1

    byol = BYOL(net=encoder, 
                    image_size=args.image_size, 
                    hidden_layer=hidden_layer, 
                    projection_size=args.feature_dim, 
                    proj_head_type=args.proj_head_type,
                    projection_hidden_size=args.proj_hidden_dim,
                    use_momentum=True).cuda()
    if args.model_path and not os.path.isdir(args.model_path) and os.path.exists(args.model_path):
      loaded_dict = torch.load(args.model_path)
      keys_to_ignore = [key for key in loaded_dict if 'online_predictor' in key]
      for key in keys_to_ignore:
        loaded_dict.pop(key)
      byol.load_state_dict(loaded_dict, strict=False)
        # byol.load_state_dict(loaded_dict, strict=True)

    model = Net(num_class=len(train_data.classes), encoder=byol.online_encoder, fname_linear_cls=args.fname_linear_cls).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    # if dataset == 'cifar10':
    #     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # elif dataset == 'tiny_imagenet' or dataset == 'stl10':
    #     flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    os.makedirs('results_linear/', exist_ok=1)
    os.makedirs('results_linear/csv', exist_ok=1)
    save_name_csv = os.path.join('results_linear/csv', args.wb_name+'_linear.csv')
    save_name_model = os.path.join('results_linear', args.wb_name+'_linear_model.pth')
    if os.path.exists(save_name_model) and not args.overwrite:
      cont = input(f"File exists: {save_name_model}. \n Continue? (y/N)")
      if 'y' not in cont and 'Y' not in cont:
        print("Exiting. Bye!")
        exit()

    best_acc = 0.0
    early_stop_cnt = 0
    test_acc_history = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)

        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(save_name_csv, index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), save_name_model)
        if USE_WANDB:
          wandb.log({
            'train_loss': train_loss,
            'linear_train_top1': train_acc_1,
            'linear_train_top5': train_acc_5,
            'test_loss': test_loss,
            'linear_test_top1': test_acc_1,
            'linear_test_top5': test_acc_5,
            })

        # check for early stop
        test_acc_history = (1-1/epoch) * test_acc_history + (1/epoch) * test_acc_1
        if epoch > 20 and test_acc_1 < test_acc_history + EARLY_STOP_EPS:
          early_stop_cnt += 1
        else:
          early_stop_cnt = 0
        if early_stop_cnt == 10:
          print("Early stopping. Bye!")
          exit()




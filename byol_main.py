from byol import BYOL
import argparse
import os
import utils
from utils import str2bool

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile, clever_format
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pickle

try:
  import wandb
  USE_WANDB = True
except Exception as e:
  print('Exception:', e)
  print('Not using wandb. \n\n')
  USE_WANDB = False


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_tuple in train_bar:
        (pos_1, pos_2), _ = data_tuple
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        loss = net(pos_1, pos_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        bt_cnt += 1
        if USE_WANDB and bt_cnt % 20 == 0:
          wandb.log({
            'loss':loss.item(),
            })

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} bsz:{} f_dim:{} proj_hidden_sim: {} dataset: {}'.format(\
                                epoch, epochs, total_loss / total_num, batch_size, args.feature_dim, args.proj_hidden_dim, args.dataset))
    return total_loss / total_num

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
            (data1, data2), target = data_tuple
            target_bank.append(target)
            feature, _ = net(data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), return_embedding=True)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data1, data2), target = data_tuple
            data1, data2, target = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, _ = net(data1, data2, return_embedding=True)

            total_num += data1.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data1.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data1.size(0) * k, NUM_CLS, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data1.size(0), -1, NUM_CLS) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    total_top1 = total_top1 / total_num
    total_top5 = total_top5 / total_num

    if USE_WANDB:
      log_dict = {
          'total_top1': total_top1,
          'total_top5': total_top5,
        }
      wandb.log(log_dict)
    return total_top1 * 100, total_top5 * 100

def parse_args():
    parser = argparse.ArgumentParser(description='Train BYOL network')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--image_size', default=32, type=int, help='Size of image')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector (after head)')
    parser.add_argument('--proj-head-type', default='2layer', choices=['none', 'linear', '2layer'],
                        help="Type of the projector.")
    parser.add_argument('--proj_hidden_dim', default=512, type=int, help='Feature dim for the hidden layer of the proj head (matters only if proj_head_type=2layer)')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--use_pretrained_enc', type=str2bool, default=False, help="use pretrained resnet 50 encoder")
    parser.add_argument('--use_default_enc', type=str2bool, default=True, help="use default resnet 50 encoder")
    parser.add_argument('--use_seed',
                        help='Should we set a seed for this particular run?',
                        type=str2bool,
                        default=False)
    parser.add_argument('--seed',
                        help='seed to fix in torch',
                        type=int,
                        default=0)
    # optimization
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-6, type=float)

    # logging
    parser.add_argument('--project', default='nonContrastive')
    parser.add_argument('--wb-name', default='default', type=str,
                        help="Run name for wandb.")
    parser.add_argument('--fSinVals', default='', type=str,
                        help="Filename (full path) for singular value plots on feat/out.")
    parser.add_argument('--save-feats', default=0, type=int,
                        help="Whether to save features (before and after proj head).")
    parser.add_argument('--fsave-feats', default='', type=str,
                        help="Full path to the file for saving features.")
    
    # testing
    parser.add_argument('--test-only', default=0, type=int, choices=[0, 1],
                        help="If test_only, then skip the training loop.")
    parser.add_argument('--load-ckpt', default=0, type=int, choices=[0,1],
                        help="Whether to load a ckpt.")
    parser.add_argument('--pretrained-path', default='', type=str,
                        help="Full path to a pretrained ckpt.")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print(torch.__version__)
    if args.use_seed:
        torch.manual_seed(args.seed)
    dataset = args.dataset
    batch_size, epochs = args.batch_size, args.epochs
    lr, wd = args.lr, args.wd
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k

    if USE_WANDB:
      if args.wb_name != 'default':
        wandb.init(project=args.project, name=args.wb_name, config=args)
      else:
        wandb.init(project=args.project, config=args)

    # Save path
    if not os.path.exists('results'):
        os.mkdir('results')
    # save_name_pre = '{}{}_{}_{}_{}'.format(corr_neg_one_str, lmbda, feature_dim, batch_size, dataset)
    save_name_pre = args.wb_name
    save_dir = os.path.join('results/', save_name_pre)
    if not args.test_only:
      if os.path.exists(save_dir):
        print(f"Dir exists: {save_name_pre}.")
        cont = input("Continue? (y/N)")
        if cont != 'y' and cont != 'Y':
          print("Exiting.")
          exit()
      os.makedirs(save_dir, exist_ok=1)
      fargs = os.path.join(save_dir, 'args.pkl')
      with open(fargs, 'wb') as handle:
        pickle.dump(args, handle)

    # data prepare
    DATA_ROOT = '/home/bingbin/datasets/'
    if dataset == 'cifar10':
        root = os.path.join(DATA_ROOT, 'cifar10_torch')
        train_data = torchvision.datasets.CIFAR10(root=root, train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.CIFAR10(root=root, train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.CIFAR10(root=root, train=False, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
    elif dataset == 'stl10':
        # TODO: update root
        root = DATA_ROOT
        train_data = torchvision.datasets.STL10(root=root, split="train+unlabeled", \
                                                  transform=utils.StlPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.STL10(root=root, split="train", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.STL10(root=root, split="test", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
    elif dataset == 'tiny_imagenet':
        # TODO: update root
        root = DATA_ROOT
        train_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = True))
        memory_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
        test_data = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
    else:
        raise ValueError(f" Unknown dataset {dataset}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.use_default_enc:
        encoder = models.resnet50(pretrained=args.use_pretrained_enc)
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

    model = BYOL(net=encoder, 
                    image_size=args.image_size, 
                    hidden_layer=hidden_layer, 
                    projection_size=args.feature_dim, 
                    projection_hidden_size=args.proj_hidden_dim,
                    proj_head_type=args.proj_head_type,
                    use_momentum=True).cuda()
    
    if args.load_ckpt and os.path.exists(args.pretrained_path):
      print("Loading ckpt from", args.pretrained_path)
      ckpt_dict = torch.load(args.pretrained_path, map_location='cpu')
      model.load_state_dict(ckpt_dict, strict=False)
    elif not args.test_only:
      # save the init
      torch.save(model.state_dict(), 'results/{}/model_init.pth'.format(save_name_pre))

    NUM_CLS = len(memory_data.classes)

    # if dataset == 'cifar10':
    #     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), torch.randn(1, 3, 32, 32).cuda()))

    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))

    if args.test_only:
      print("\nNot yet implemented.\n")
      pdb.set_trace()
      # plots
      fig_dir = os.path.dirname(args.fSinVals)
      os.makedirs(fig_dir, exist_ok=1)
      # features
      save_feats = args.save_feats
      fsave_feats = args.fsave_feats
      feat_dir = os.path.dirname(fsave_feats)
      os.makedirs(feat_dir, exist_ok=1)

      print("On test set")
      fSinVals_test = args.fSinVals + '_test.png'
      fsave_feats_test = args.fsave_feats + '_test.h5'
      test_stats(model, test_loader, fSinVals=fSinVals_test, save_feats=save_feats, fsave_feats=fsave_feats_test)
      exit()
      print("On train set")
      fSinVals_train = args.fSinVals + '_train.png'
      fsave_feats_train = args.fsave_feats + '_train.h5'
      test_stats(model, memory_loader, fSinVals=fSinVals_train, save_feats=save_feats, fsave_feats=fsave_feats_train)
      exit()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        if epoch % 5 == 0:
            results['train_loss'].append(train_loss)
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(5, epoch + 1, 5))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), 'results/{}_model_{}.pth'.format(save_name_pre, epoch))

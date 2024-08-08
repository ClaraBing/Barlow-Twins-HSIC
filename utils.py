from PIL import Image
from torchvision import transforms
# from torchvision.datasets import CIFAR10

# for cifar10 (32x32)
class CifarPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# for tiny imagenet (64x64)
class TinyImageNetPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
                ])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)

# for stl10 (96x96)
class StlPairTransform:
    def __init__(self, train_transform = True, pair_transform = True):
        if train_transform is True:
            self.transform = transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                saturation=0.4, hue=0.1)], 
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomResizedCrop(
                        64,
                        scale=(0.2, 1.0),
                        ratio=(0.75, (4 / 3)),
                        interpolation=Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(70, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))
                ])
        self.pair_transform = pair_transform
    def __call__(self, x):
        if self.pair_transform is True:
            y1 = self.transform(x)
            y2 = self.transform(x)
            return y1, y2
        else:
            return self.transform(x)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_stable_ranks(ffeats):
  with h5py.File(ffeats, "r") as f:
    outs = np.array(f['outs'])
    feats = np.array(f['feats'])

  _, ss_outs, _ = np.linalg.svd(outs)
  stable_rank_outs = ss_outs.sum() / ss_outs.max()
  _, ss_feats, _ = np.linalg.svd(feats)
  stable_rank_feats = ss_feats.sum() / ss_feats.max()
  print('Stable ranks for', ffeats)
  print('lF/l2:\t outs: {:.3f} / feats: {:.3f}'.format(stable_rank_outs, stable_rank_feats))
  print('(lF/l2)^2:\t outs: {:.3f} / feats: {:.3f}'.format(stable_rank_outs**2, stable_rank_feats**2))

def check_feats_vs_cls(ffeats, fcls):
  with h5py.File(ffeats, "r") as f:
    # shape: n_pts x feat_dim
    feats = np.array(f['feats'])
  loaded_sd = torch.load(fcls)
  # shape: n_cls x feat_dim
  clsW = loaded_sd['fc.weight'].cpu().numpy()

  Uf, ssf, Vf = np.linalg.svd(feats, full_matrices=0)
  Uc, ssc, Vc = np.linalg.svd(clsW, full_matrices=0)
  prod = (np.diag(ssc).dot(Vc)).dot((np.diag(ssf).dot(Vf)).T)
  _, ss_prod, _ = np.linalg.svd(prod)

  stable_rank_feats = ssf.sum() / ssf.max()
  stable_rank_cls = ssc.sum() / ssc.max()
  stable_rank_prod = ss_prod.sum() / ss_prod.max()
  trace_prod = ss_prod.sum()

  print(f"Stable rank for:\n\t {os.path.basename(ffeats)}\n\t {os.path.basename(fcls)}")
  print(f"Feats: stable rank: {stable_rank_feats:.2e} / smax:{ssf.max():.2e} / smin:{ssf.min():.2e}")
  print(f"Cls: stable rank: {stable_rank_cls:.2e} / smax:{ssc.max():.2e} / smin:{ssc.min():.2e}")
  print(f'stable_rank_prod: {stable_rank_prod:.2e} / smax:{ss_prod.max():.2e} / smin:{ss_prod.min():.2e}')
  print(f"trace prod: {trace_prod:.2e}")
  print()
  pdb.set_trace()

if __name__ == '__main__':
  import os
  import h5py
  import numpy as np
  from glob import glob
  import torch
  import pdb

  if 0:
    # check the stable ranks for all features
    ffeats_list = glob('saved_feats/*')
    print(f"{len(ffeats_list)} features to check.")
    for ffeats in ffeats_list:
      if '0.005' not in ffeats:
        continue
      check_stable_ranks(ffeats)
      print()

  if 1:
    ffeats_lst = [
      'dim128_lmbda0.005_bt128_sameInit2_test.h5', # lmbda=0.005
      'dim128_lmbda0.05_bt128_test.h5', # lmbda=0.05
      'byol_dim128_lmbda0.005_bt128_test_Ashwini.h5', # BYOL
    ]
    fcls_lst = [
      'cifar10_linear_linear_feat128_lmbda0.005_lr1e-4_wd1e-5_bt128_sameInit2_linear_model.pth', # lmbda=0.005
      'cifar10_linear_linear_feat128_lmbda0.05_lr3e-4_wd1e-5_bt128_linear_model.pth', # lmbda=0.05
      'byol_cifar10_linear_linear_noBNReLU_feat512_lr1e-3_wd1e-5_bt128_linear_model.pth' # BYOL
    ]
    for ffeats in ffeats_lst:
      for fcls in fcls_lst:
        ffeats_full = os.path.join('saved_feats', ffeats)
        fcls_full = os.path.join('results_linear', fcls)
        check_feats_vs_cls(ffeats_full, fcls_full)
        print('\n\n')




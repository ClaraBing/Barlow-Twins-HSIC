import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# from seaborn import heatmap

from sklearn.cross_decomposition import CCA

import pdb


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device='cpu'
print(f"Using device {device}")

VERBOSE = 0
VERBOSE_TRAIN = 0
FIG_DIR='figs/align/'


def train_gd(max_num_epochs, X, Y, optimizer_type, lr, momentum, log_interval=10,
      l1_reg=0, l2_reg=0, abs_reg=0):
    if VERBOSE_TRAIN:
      print(f"\n optimizer_type=={optimizer_type}, lr=={lr}, momentum=={momentum}\n")

    assert X.shape == Y.shape
    n, d = X.shape
    X = torch.tensor(X, device=device, requires_grad=False)
    Y = torch.tensor(Y, device=device, requires_grad=False)
    
    A = torch.randn((d, d), device=device, dtype=torch.float) / d**0.5
    A.requires_grad = True
    b = torch.randn((d, ), device=device, dtype=torch.float, requires_grad=True)
    A_optimal = A.detach().cpu().numpy()
    b_optimal = b.detach().cpu().numpy()

    # pdb.set_trace()

    loss_history = []
    if momentum is None:
        optimizer = optimizer_type([A, b], lr=lr)
    else:
        optimizer = optimizer_type([A, b], lr=lr, momentum=momentum)
    for epoch in range(max_num_epochs):
        optimizer.zero_grad()
        loss = torch.norm(X @ A + b - Y)
        loss_history.append(float(loss))

        if l1_reg > 0:
          loss = loss + l1_reg * (torch.norm(A, 1) + torch.norm(b, 1))
        if l2_reg > 0:
          loss = loss + l2_reg * (torch.norm(A) + torch.norm(b))
        if abs_reg > 0:
          loss = loss + abs_reg * (A.abs().sum() + b.abs().sum())
        if epoch % log_interval == 0 and VERBOSE_TRAIN:
            print(f"Epoch {epoch}, loss {loss}")
        if len(loss_history) >= 1 and loss < min(loss_history):
            A_optimal = A.detach().cpu().numpy()
            b_optimal = b.detach().cpu().numpy()
        loss.backward()
        optimizer.step()
    
    # pdb.set_trace()
    return loss_history, A_optimal, b_optimal



def plot_results(A_optimal, b_optimal, fname='align_sVals.png'):
    # Inspect the optimal A and b
    u, s, vh = np.linalg.svd(A_optimal, full_matrices=True)
    if VERBOSE:
      print('Singular values', s)
    print('Stable rank (l2/lF):', s.max() / s.sum())
    
    plt.figure()
    plt.plot(range(len(s)), s)
    plt.savefig(fname)
    plt.clf()

    plt.figure()
    plt.plot(range(len(s[:20])), s[:20])
    plt.savefig(fname.replace('.png', '_top20.png'))
    plt.clf()

    print('b_norm', np.linalg.norm(b_optimal))


def check_alignment(features_file1, features_file2, layer_key, fname='',
      method='GD', l1_reg=0, l2_reg=0, abs_reg=0):
    assert layer_key in ['outs', 'feats']  # 'outs' is 128-dim, 'feats' means 2048-dim

    with h5py.File(features_file1, "r") as f:
        features1 = np.array(f[layer_key])

    with h5py.File(features_file2, "r") as f:
        features2 = np.array(f[layer_key])
    unaligned_loss = np.linalg.norm(features1 - features2)
    
    if method == 'CCA':
      # CCA
      cca = CCA(features1.shape[1])
      cca.fit(features1, features2)
      X, Y = cca.transform(features1, features2)
      loss = np.linalg.norm(X - Y)
      print('Unaligned loss:\t', unaligned_loss)
      print('loss (CCA):\t', loss)
      pdb.set_trace()

    elif method == 'GD':
      # GD: Try multiple optimizer settings
      optimizer_settings = []
      for lr in [0.3, 0.1, 0.03, 0.001]:
          for momentum in [0.9, 0.3, 0.1, 0.0]:
              optimizer_settings.append((torch.optim.SGD, lr, momentum))
      for lr in [0.3, 0.1, 0.03, 0.001]:
          optimizer_settings.append((torch.optim.Adam, lr, None))

      results = {}
      max_num_epochs = 200
      for optimizer_type, lr, momentum in optimizer_settings:
          loss_history, A_optimal, b_optimal = train_gd(max_num_epochs, features1, features2, optimizer_type, lr, momentum,
              log_interval=20, l1_reg=l1_reg, l2_reg=l2_reg, abs_reg=abs_reg)
          results[(optimizer_type, lr, momentum)] = {
                  'min_loss': min(loss_history),
                  'num_epochs': len(loss_history),
                  'A_optimal': A_optimal,
                  'b_optimal': b_optimal,
              }
      
      # Select the best
      min_loss = float("inf")
      best_hyperparam = None
      A_optimal, b_optimal = (None, None)

      for hyperparam in results.keys():
          result = results[hyperparam]
          if result['min_loss'] < min_loss:
              min_loss = result['min_loss']
              A_optimal = result['A_optimal']
              b_optimal = result['b_optimal']
              best_hyperparam = hyperparam 

      print('best_hyperparam: (optimizer_type, lr, momentum) = ', best_hyperparam)
      print('Unaligned loss:\t', unaligned_loss)
      print('min_loss:\t', min_loss)
      if VERBOSE:
        print('A_optimal', A_optimal)
        print('b_optimal', b_optimal)

      if fname:
        plot_results(A_optimal, b_optimal, fname)

def May11():
  if 0:
    # # lambda = 0.05 diff init
    f1 = 'saved_feats/dim128_lmbda0.05_bt128_test.h5' 
    f2 = 'saved_feats/dim128_lmbda0.05_bt128_diffInit_test.h5' 
    fname = os.path.join(FIG_DIR, 'bt_lmdba0.05_diffInits.png')
  
  if 1:
    # BYOL (1st run) vs BYOL (2nd run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test_Ashwini.h5'
    f2 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test.h5'
    # NOTE: this run is saved with the wrong name (with '_diffInits')
    fname = os.path.join(FIG_DIR, 'byol1_vs_byol2.png')

  if 0:
    # BYOL (1st run) vs BT=0.05 (1st run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test_Ashwini.h5'
    f2 = 'saved_feats/dim128_lmbda0.05_bt128_test.h5' 
    # NOTE: this run is saved with the wrong name (with '_diffInits')
    fname = os.path.join(FIG_DIR, 'byol1_vs_BTlmdba0.05.png')

  if 0:
    # BYOL (1st run) vs BT=0.05 (2nd run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test_Ashwini.h5'
    f2 = 'saved_feats/dim128_lmbda0.05_bt128_diffInit_test.h5' 
    fname = os.path.join(FIG_DIR, 'byol1_vs_BTlmdba0.05_diffInits.png')

  if 0:
    # BYOL (2nd run) vs BT=0.05 (1st run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test.h5'
    f2 = 'saved_feats/dim128_lmbda0.05_bt128_test.h5' 
    fname = os.path.join(FIG_DIR, 'byol2_vs_BTlmdba0.05.png')

  if 0:
    # BYOL (2nd run) vs BT=0.05 (2nd run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test.h5'
    f2 = 'saved_feats/dim128_lmbda0.05_bt128_diffInit_test.h5' 
    fname = os.path.join(FIG_DIR, 'byol2_vs_BTlmdba0.05_diffInits.png')

  if 0:
    # BYOL (1st run) vs BT=0.005 (1st run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test_Ashwini.h5'
    f2 = 'saved_feats/dim128_lmbda0.005_bt128_sameInit_test.h5' 
    fname = os.path.join(FIG_DIR, 'byol1_vs_BTlmdba0.005.png')

  if 0:
    # BYOL (2nd run) vs BT=0.005 (1st run)
    f1 = 'saved_feats/byol_dim128_lmbda0.005_bt128_test.h5'
    f2 = 'saved_feats/dim128_lmbda0.005_bt128_sameInit_test.h5' 
    fname = os.path.join(FIG_DIR, 'byol2_vs_BTlmdba0.005.png')

  l1_reg = 0 
  l2_reg = 0
  abs_reg = 0
  if l1_reg > 0:
    fname = fname.replace('.png', f'_l1{l1_reg}.png')
  if l2_reg > 0:
    fname = fname.replace('.png', f'_l2{l2_reg}.png')
  if abs_reg > 0:
    fname = fname.replace('.png', f'_abs{abs_reg}.png')
  print(fname.replace('.png', ''))

  method = 'CCA'

  # ## 128-dim
  print("Outs (dim 128)")
  check_alignment(f1, f2, 'outs', fname=fname, l1_reg=l1_reg, l2_reg=l2_reg, abs_reg=abs_reg, method=method)

  # ## 2048-dim
  # print("\nFeats (dim 2048)")
  # check_alignment(f1, f2, 'feats', fname=fname, l1_reg=l1_reg, l2_reg=l2_reg, abs_reg=abs_reg, method=method)

if 0:
  # # lambda = 0.005 same init different runs
  
  # ## 128-dim
  check_alignment(
          'output/dim128_lmbda0.005_bt128_sameInit_test.h5', 
          'output/dim128_lmbda0.005_bt128_sameInit2_test.h5', 
          'outs',
      )
  # ## 2048-dim
  check_alignment(
          'output/dim128_lmbda0.005_bt128_sameInit_test.h5', 
          'output/dim128_lmbda0.005_bt128_sameInit2_test.h5', 
          'feats',
      )

if 0:
  # # lambda = 0.005 diff init
  # ## 128-dim
  check_alignment(
          'output/dim128_lmbda0.005_bt128_sameInit_test.h5', 
          'output/dim128_lmbda0.005_bt128_test.h5', 
          'outs',
  )
    
  # ## 2048-dim
  check_alignment(
          'output/dim128_lmbda0.005_bt128_sameInit_test.h5', 
          'output/dim128_lmbda0.005_bt128_test.h5', 
          'feats',
  )
  

if 0:
  # # lambda = 0.005 vs. lambda = 0.05
  # ## 128-dim
  check_alignment(
          'output/dim128_lmbda0.005_bt128_test.h5', 
          'output/dim128_lmbda0.05_bt128_test.h5', 
          'outs',
      )
  # ## 2048-dim
  check_alignment(
          'output/dim128_lmbda0.005_bt128_test.h5', 
          'output/dim128_lmbda0.05_bt128_test.h5', 
          'feats',
      )


if 0:
  # # BYOL vs. lambda = 0.05
  # ## 128-dim
  check_alignment(
          'output/byol_dim128_test.h5', 
          'output/dim128_lmbda0.05_bt128_test.h5', 
          'outs',
      )
  # ## 2048-dim
  check_alignment(
          'output/byol_dim128_test.h5', 
          'output/dim128_lmbda0.05_bt128_test.h5', 
          'feats',
      )

if 0:
  # # Comparison: linear alignment between random matrices
  # ## 128-dim
  num_trials = 5
  min_losses = []
  for i in range(num_trials):
      X = torch.randn((10000, 128), device=device, dtype=torch.float, requires_grad=False)
      Y = torch.randn((10000, 128), device=device, dtype=torch.float, requires_grad=False)
      loss_history, A_optimal, b_optimal = train_gd(100, X, Y, torch.optim.SGD, lr=0.01, momentum=0.9)
      min_losses.append(min(loss_history))
  
  print('min_losses', min_losses)
  avg_min_loss = sum(min_losses) / len(min_losses)
  print('avg_min_loss', avg_min_loss)
  plot_results(A_optimal, b_optimal)
 
  # ## 2048-dim
  num_trials = 3
  min_losses = []
  for i in range(num_trials):
      X = torch.randn((10000, 2048), device=device, dtype=torch.float, requires_grad=False)
      Y = torch.randn((10000, 2048), device=device, dtype=torch.float, requires_grad=False)
      loss_history, A_optimal, b_optimal = train_gd(100, X, Y, torch.optim.SGD, lr=0.01, momentum=0.9)
      min_losses.append(min(loss_history))
  
  print('min_losses', min_losses)
  avg_min_loss = sum(min_losses) / len(min_losses)
  print('avg_min_loss', avg_min_loss)
  plot_results(A_optimal, b_optimal)


if 0:
  # # Check the moments of features and random baselines
  features_file1 = 'output/dim128_lmbda0.005_bt128_test.h5'
  features_file2 = 'output/dim128_lmbda0.05_bt128_test.h5'
  byol_file = 'output/byol_dim128_test.h5'
  with h5py.File(features_file1, "r") as f:
      outs1 = np.array(f['outs'])
      feats1 = np.array(f['feats'])
  with h5py.File(byol_file, "r") as f:
      byol_outs1 = np.array(f['outs'])
      byol_feats1 = np.array(f['feats'])
  print(outs1.shape, feats1.shape)

  X128 = torch.randn((10000, 128), device=device, dtype=torch.float, requires_grad=False).to('cpu').numpy()
  X2048 = torch.randn((10000, 2048), device=device, dtype=torch.float, requires_grad=False).to('cpu').numpy()
  
  # ## Mean, 128-dim
  np.mean(outs1, axis=0)
  np.mean(X128, axis=0)
  np.mean(byol_outs1, axis=0)
  
  # ## Mean, 2048-dim
  print(np.mean(feats1, axis=0))
  print(min(np.mean(feats1, axis=0)))
  print(max(np.mean(feats1, axis=0)))
  
  print(np.mean(X2048, axis=0))
  print(min(np.mean(X2048, axis=0)))
  print(max(np.mean(X2048, axis=0)))
  
  print(np.mean(byol_feats1, axis=0))
  print(min(np.mean(byol_feats1, axis=0)))
  print(max(np.mean(byol_feats1, axis=0)))

  # ## Cov, 128-dim
  np.cov(outs1, rowvar=False)
  np.cov(X128, rowvar=False)
  np.cov(byol_outs1, rowvar=False)
  
  # ## Cov, 2048-dim
  np.cov(feats1, rowvar=False)
  np.cov(X2048, rowvar=False)
  np.cov(byol_feats1, rowvar=False)

if __name__ == '__main__':
  May11()


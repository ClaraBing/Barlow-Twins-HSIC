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


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device='cpu'
print(f"Using device {device}")

FIG_DIR='figs/align/'

def train_gd(max_num_epochs, X, Y, optimizer_type, lr, momentum, log_interval=10):
    print(f"\n optimizer_type=={optimizer_type}, lr=={lr}, momentum=={momentum}\n")

    assert X.shape == Y.shape
    n, d = X.shape
    X = torch.tensor(X, device=device, requires_grad=False)
    Y = torch.tensor(Y, device=device, requires_grad=False)
    
    A = torch.randn((d, d), device=device, dtype=torch.float, requires_grad=True)
    b = torch.randn((d, ), device=device, dtype=torch.float, requires_grad=True)
    A_optimal = A.detach().cpu().numpy()
    b_optimal = b.detach().cpu().numpy()

    loss_history = []
    if momentum is None:
        optimizer = optimizer_type([A, b], lr=lr)
    else:
        optimizer = optimizer_type([A, b], lr=lr, momentum=momentum)
    for epoch in range(max_num_epochs):
        optimizer.zero_grad()
        loss = torch.norm(X @ A + b - Y)
        if epoch % log_interval == 0:
            print(f"Epoch {epoch}, loss {loss}")
        if len(loss_history) >= 1 and loss < min(loss_history):
            A_optimal = A.detach().cpu().numpy()
            b_optimal = b.detach().cpu().numpy()
        loss_history.append(float(loss))
        loss.backward()
        optimizer.step()
    
    return loss_history, A_optimal, b_optimal



def plot_results(A_optimal, b_optimal, fname='align_sVals.png'):
    # Inspect the optimal A and b
    u, s, vh = np.linalg.svd(A_optimal, full_matrices=True)
    print('Singular values', s)
    
    plt.figure()
    plt.plot(range(len(s)), s)
    plt.savefig()
    plt.clf()

    plt.figure()
    plt.plot(range(len(s[:20])), s[:20])
    plt.savefig(fname.replace('.png', '_top20.png'))
    plt.clf()

    print('b_norm', np.linalg.norm(b_optimal))


def check_alignment(features_file1, features_file2, layer_key, fname=''):
    assert layer_key in ['outs', 'feats']  # 'outs' is 128-dim, 'feats' means 2048-dim

    with h5py.File(features_file1, "r") as f:
        features1 = np.array(f[layer_key])

    with h5py.File(features_file2, "r") as f:
        features2 = np.array(f[layer_key])
    
    # Try multiple optimizer settings
    optimizer_settings = []
    for lr in [0.3, 0.1, 0.03, 0.001]:
        for momentum in [0.9, 0.3, 0.1, 0.0]:
            optimizer_settings.append((torch.optim.SGD, lr, momentum))
    for lr in [0.3, 0.1, 0.03, 0.001]:
        optimizer_settings.append((torch.optim.Adam, lr, None))

    results = {}
    max_num_epochs = 200
    for optimizer_type, lr, momentum in optimizer_settings:
        loss_history, A_optimal, b_optimal = train_gd(max_num_epochs, features1, features2, optimizer_type, lr, momentum, log_interval=20)
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

    print('min_loss', min_loss)
    print('best_hyperparam: (optimizer_type, lr, momentum) = ', best_hyperparam)
    print('A_optimal', A_optimal)
    print('b_optimal', b_optimal)

    if fname:
      plot_results(A_optimal, b_optimal, fname)


if 1:
  # # lambda = 0.05 diff init
  f1 = 'saved_feats/dim128_lmbda0.05_bt128_test.h5' 
  f2 = 'saved_feats/dim128_lmbda0.05_bt128_diffInit_test.h5' 
  
  fname = os.path.join(FIG_DIR, 'bt_lmdba0.05_diffInits.png')
  # ## 128-dim
  check_alignment(f1, f2, 'outs', fname=fname)
  # ## 2048-dim
  check_alignment(f1, f2, 'feats', fname=fname)

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


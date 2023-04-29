import math
import numpy as np
import torch
import gpytorch
import random
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
#sys.path.append("../directionalvi")
sys.path.append("../utils")

from RBFKernelDirectionalGrad import RBFKernelDirectionalGrad #.RBFKernelDirectionalGrad
from DirectionalGradVariationalStrategy import DirectionalGradVariationalStrategy #.DirectionalGradVariationalStrategy
from count_params import count_params


"""Future Upgrades
- don't include function values in every training iteration... be truly stochastic.
"""

class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self,inducing_points,inducing_directions,learn_inducing_locations=True,**kwargs):

        self.num_inducing   = len(inducing_points)
        self.num_directions = int(len(inducing_directions)/self.num_inducing) # num directions per point
        num_directional_derivs = self.num_directions*self.num_inducing

        # variational distribution q(u,g)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
          self.num_inducing + num_directional_derivs
        )

        # variational strategy
        variational_strategy = DirectionalGradVariationalStrategy(
          self,
          inducing_points,
          inducing_directions,
          variational_distribution, 
          learn_inducing_locations=learn_inducing_locations
        )
        super(GPModel, self).__init__(variational_strategy)

        # set the mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernelDirectionalGrad())

    def forward(self, x, **params):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x, **params)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def select_cols_of_y(y_batch,minibatch_dim,dim):
  """
  randomly select columns of y to train on, but always select 
  function values as part of the batch. Otherwise we have
  to keep track of whether we passed in function values or not
  when computing the kernel.

  input
  y_batch: 2D-torch tensor
  minibatch_dim: int, total number of derivative columns of y to sample
  dim: int, problem dimension
  """
  # randomly select columns of y to train on
  idx_y   = random.sample(range(1,dim+1),minibatch_dim) # ensures unique entries
  idx_y  += [0] # append 0 to the list for function values
  idx_y.sort()
  y_batch = y_batch[:,idx_y]

  # dont pass a direction if we load function values
  # E_canonical = torch.eye(dim).to(y_batch.device)
  # derivative_directions = E_canonical[np.array(idx_y[1:])-1]
  derivative_directions = torch.zeros((minibatch_dim, dim))
  for i in range(minibatch_dim):
    y_selected = np.array(idx_y[i+1])-1
    derivative_directions[i, y_selected] = 1

  return y_batch,derivative_directions

def init_gp(dim, num_inducing, num_directions, inducing_points=None):
  if inducing_points is None:
    inducing_points = torch.rand(num_inducing, dim)

  inducing_directions = torch.zeros((num_directions, dim)) 
  for i in range(num_directions):
    inducing_directions[i,i] = 1
  inducing_directions = inducing_directions.repeat(num_inducing,1)

  # initialize model
  model = GPModel(inducing_points,inducing_directions)
  likelihood = gpytorch.likelihoods.GaussianLikelihood()

  return model, likelihood

def train_gp(
  model,
  likelihood,
  train_x, 
  train_y,
  num_directions=1,
  minibatch_size=1024,
  num_epochs=10,
  learning_rate=0.01,
  mll_type="ELBO",
  gamma=0.2,
  verbose=True,
  ):
  """Train a Derivative GP with the Directional Derivative
  Variational Inference method

  train_dataset: torch Dataset
  num_directions: int, number of inducing directions (per inducing point)
  minbatch_size: int, number of data points in a minibatch
  minibatch_dim: int, number of derivative per point in minibatch training
                 WARNING: This must equal num_directions until we complete
                 the PR in GpyTorch.
  num_epochs: int, number of epochs
  inducing_data_initialization: initialize the inducing points as a set of 
      data points. If False, the inducing points are generated on the unit cube
      uniformly, U[0,1]^d.
  learning_rate_hypers, float: initial learning rate for the hyper optimizer
  learning_rate_ngd, float: initial learning rate for the variational optimizer
  use_ngd, bool: use NGD
  use_ciq, bool: use CIQ
  lr_sched, function handle: used in the torch LambdaLR learning rate scheduler. At
      each iteration the initial learning rate is multiplied by the result of 
      this function. The function input is the epoch, i.e. lr_sched(epoch). 
      The function should return a single number. If lr_sched is left as None, 
      the learning rate will be held constant.
  """
  minibatch_dim = num_directions

  # set up the data loader
  train_dataset = TensorDataset(train_x, train_y)
  train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
  dim = len(train_dataset[0][0])
  n_samples = len(train_dataset)
  num_data = (dim+1)*n_samples

  if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
  
  # training mode
  model.train()
  likelihood.train()

  # optimizers
  variational_optimizer = torch.optim.Adam([
      {'params': model.variational_parameters()},
  ], lr=learning_rate)
  hyperparameter_optimizer = torch.optim.Adam([
      {'params': model.hyperparameters()},
      {'params': likelihood.parameters()},
  ], lr=learning_rate)
      
  # learning rate scheduler
  num_batches = int(np.ceil(n_samples/minibatch_size))
  milestones = [int(num_epochs*num_batches/3), int(2*num_epochs*num_batches/3)]
  hyperparameter_scheduler = torch.optim.lr_scheduler.MultiStepLR(hyperparameter_optimizer, milestones, gamma=gamma)
  variational_scheduler = torch.optim.lr_scheduler.MultiStepLR(variational_optimizer, milestones, gamma=gamma)

  # mll
  if mll_type=="ELBO":
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
  elif mll_type=="PLL": 
    mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=num_data)


  total_step=0
  for i in range(num_epochs):
    # iterator for minibatches
    minibatch_iter = train_loader
    # loop through minibatches
    for x_batch, y_batch in minibatch_iter:
      if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

      # select random columns of y_batch to train on
      y_batch,derivative_directions = select_cols_of_y(y_batch,minibatch_dim,dim)
      kwargs = {}
      # repeat the derivative directions for each point in x_batch
      kwargs['derivative_directions'] = derivative_directions.repeat(y_batch.size(0),1)

      # pass in interleaved data... so kernel should also interleave
      y_batch = y_batch.reshape(torch.numel(y_batch))

      variational_optimizer.zero_grad()
      hyperparameter_optimizer.zero_grad()
      output = likelihood(model(x_batch,**kwargs))
      loss = -mll(output, y_batch)
      loss.backward()
      # step optimizers and learning rate schedulers
      variational_optimizer.step()
      variational_scheduler.step()
      hyperparameter_optimizer.step()
      hyperparameter_scheduler.step()
      if total_step % 50 == 0 and verbose:
          means = output.mean[::num_directions+1]
          stds  = output.variance.sqrt()[::num_directions+1]
          nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch[::num_directions+1]).mean()
          print(f"Epoch: {i}; total_step: {total_step}, loss: {loss.item()}, nll: {nll}")
          sys.stdout.flush()

      total_step +=1
     
  if verbose:
    print(f"Done! loss: {loss.item()}")

  return model,likelihood


def eval_gp(
  model,
  likelihood,
  test_x, 
  test_y,
  num_directions=1,
  ):

  test_dataset = TensorDataset(test_x, test_y)
  test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
  dim = len(test_dataset[0][0])  
  
  model.eval()
  likelihood.eval()
  
  kwargs = {}
  means = torch.tensor([0.])
  variances = torch.tensor([0.])
  with torch.no_grad():
    for x_batch, y_batch in test_loader:
      if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
      # redo derivative directions b/c batch size is not consistent
      derivative_directions = torch.eye(dim)[:num_directions]
      derivative_directions = derivative_directions.repeat(len(x_batch),1)
      kwargs['derivative_directions'] = derivative_directions
      # predict
      preds = likelihood(model(x_batch,**kwargs))
      means = torch.cat([means, preds.mean.cpu()])
      variances = torch.cat([variances, preds.variance.cpu()])

  means = means[1:]
  variances = variances[1:]

  return means, variances


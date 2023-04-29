from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
import torch
import sys
import gpytorch
import numpy as np



class GPModel(ApproximateGP):
    def __init__(self, inducing_points,**kwargs):
    
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def init_gp(dim, num_inducing, inducing_points=None):
    if inducing_points is None:
        inducing_points = torch.rand(num_inducing, dim)
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    return model, likelihood

def train_gp(
    model,
    likelihood,
    train_x, 
    train_y,
    minibatch_size=1024,
    num_epochs=10,
    learning_rate=0.01,
    mll_type="ELBO",
    gamma=0.2,
    verbose=True,
):
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)

    n_samples = len(train_dataset)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

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
    
    # Our loss object. We're using the VariationalELBO
    if mll_type=="ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_samples)
    elif mll_type=="PLL": 
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=n_samples)
    
    epochs_iter = range(num_epochs)
    total_step=0
    for i in epochs_iter:
        minibatch_iter = train_loader

        for x_batch, y_batch in minibatch_iter:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = likelihood(model(x_batch))
            loss = -mll(output, y_batch)
            loss.backward()
            # step optimizers and learning rate schedulers
            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()

            if total_step % 50 == 0 and verbose:
                means = output.mean
                stds  = output.variance.sqrt()
                nll   = -torch.distributions.Normal(means, stds).log_prob(y_batch).mean()
                print(f"Epoch: {i}; total_step: {total_step}, loss: {loss.item()}, nll: {nll}")
            total_step +=1
            sys.stdout.flush()
        
    if verbose:
        print(f"Done! loss: {loss.item()}")
    sys.stdout.flush()
    return model, likelihood

def eval_gp(model,likelihood,test_x,test_y):
  
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model.eval()
    likelihood.eval()
    
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            preds = likelihood(model(x_batch))
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
    means = means[1:]
    variances = variances[1:]

    return means, variances

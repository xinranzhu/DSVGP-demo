import time
import random
import torch
import gpytorch
import sys
sys.path.append("../utils/")
sys.path.append("../models/")
from metrics import MSE
from svgp import init_gp, train_gp, eval_gp
import testfun


# setups
train_n  = 600
test_n = 1000
dim = 2
num_inducing = 20
num_epochs = 400
learning_rate=0.01
gamma = 0.2
mll_type = "ELBO"
verbose = True
torch.random.manual_seed(0)

# trainig and testing data
train_x = torch.rand(train_n,dim)
test_x = torch.rand(test_n,dim)
train_y = testfun.f(train_x, deriv=False)
test_y = testfun.f(test_x, deriv=False)
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

print("\n\n---Standard SVGP---")
print(f"Start training with {train_n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points")


# initialize SVGP model 
rand_index = random.sample(range(train_n), num_inducing)
inducing_points = train_x[rand_index, :]
model, likelihood = init_gp(dim, num_inducing, inducing_points=inducing_points)

# model training
t1 = time.time_ns()	
model,likelihood = train_gp(
    model,
    likelihood,
    train_x,
    train_y,  
    mll_type=mll_type,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    gamma=gamma,
    verbose=verbose,
    )
t2 = time.time_ns()	

# model evaluation (prediction on test set)
means, variances = eval_gp(
    model,
    likelihood, 
    test_x,
    test_y,
)
t3 = time.time_ns()	

# tesing metric MSE
test_mse = MSE(test_y.cpu(),means)
# testing metric mean negative predictive density
test_nll = -torch.distributions.Normal(means, variances.sqrt()).log_prob(test_y.cpu()).mean()
print(f"At {test_n} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}")
print(f"Training time: {(t2-t1)/1e9:.2f} sec, testing time: {(t3-t2)/1e9:.2f} sec")

# TODO: call plot_testfun to plot the results

import torch
import gpytorch
import random
import time
from matplotlib import pyplot as plt
import sys
sys.path.append("../models/")
sys.path.append("../utils")
from dsvgp import init_gp, train_gp, eval_gp
from metrics import MSE
import testfun

train_n = 600
test_n = 1000
dim = 2
num_inducing = 20
num_directions = 2
num_epochs = 400
learning_rate = 0.01
gamma  = 0.2
mll_type = "ELBO"
verbose = True
torch.random.manual_seed(0)


# training and testing data
train_x = torch.rand(train_n,dim)
test_x = torch.rand(test_n,dim)
train_y = testfun.f(train_x, deriv=True)
test_y = testfun.f(test_x, deriv=True)
if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

# initialize model 
# initialize inducing points and directions from data
rand_index = random.sample(range(train_n), num_inducing)
inducing_points = train_x[rand_index, :]

# initialize model
model, likelihood = init_gp(dim, num_inducing, num_directions, inducing_points=inducing_points)

# train
print("\n\n---DirectionalGradVGP---")
print(f"Start training with {train_n} trainig data of dim {dim}")
print(f"VI setups: {num_inducing} inducing points, {num_directions} inducing directions")
args={"verbose":True}
t1 = time.time()	
model,likelihood = train_gp(
    model,
    likelihood,
    train_x,
    train_y,
    num_directions=num_directions,
    num_epochs=num_epochs, 
    learning_rate=learning_rate,
    verbose=verbose,
)
t2 = time.time()	

# save the model
# torch.save(model.state_dict(), "../data/test_dvi_basic.model")

# test
means, variances = eval_gp( 
    model,
    likelihood,
    test_x,
    test_y,
    num_directions=num_directions,
)
t3 = time.time()	

# compute MSE
test_y = test_y.cpu()
test_mse = MSE(test_y[:,0],means[::num_directions+1])
# compute mean negative predictive density
test_nll = -torch.distributions.Normal(means[::num_directions+1], variances.sqrt()[::num_directions+1]).log_prob(test_y[:,0]).mean()
print(f"At {test_n} testing points, MSE: {test_mse:.4e}, nll: {test_nll:.4e}.")
print(f"Training time: {(t2-t1):.2f} sec, testing time: {(t3-t2):.2f} sec")

# TODO: call plot_testfun to plot the results



plot=0
if plot == 1:
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y[:,0], color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means[::num_directions+1], color='b')
    plt.title("f(x,y) variational fit; actual curve is black, variational is blue")
    plt.show()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y[:,1], color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means[1::num_directions+1], color='b')
    plt.title("df/dx variational fit; actual curve is black, variational is blue")
    plt.show()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_x[:,0],test_x[:,1],test_y[:,2], color='k')
    ax.scatter(test_x[:,0],test_x[:,1],means[2::num_directions+1], color='b')
    plt.title("df/dy variational fit; actual curve is black, variational is blue")
    plt.show()

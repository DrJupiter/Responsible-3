## Adding path (for some it is needed to import packages)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Saving and loading
import pickle

## Jax

# Stop jax from taking up 90% of GPU vram
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.5'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

import jax
import jax.numpy as jnp

# Data
from data import dataload 

# TODO: Discuss this design choice in terms of the optimizer
# maybe make this a seperate module

# Model and optimizer
from models.model import get_model 
from optimizers.optimizer import get_optim

# Loss
from loss.loss import get_loss
    
# Visualization
from visualization.visualize import display_images

## Weights and biases
import wandb

# config mangement
import hydra

## Optimizer
import optax

### Train loop:

# Gets config
@hydra.main(config_path="configs/", config_name="defaults", version_base='1.3')
def run_experiment(cfg):

    # initialize Weights and Biases
    print(cfg)
    wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)

    # Get randomness key
    key = jax.random.PRNGKey(cfg.model.key)
    key, subkey = jax.random.split(key)

    # Load train and test sets
    train_dataset, test_dataset = dataload(cfg) 

    # Get model forward call and its parameters
    model_parameters, model_call = get_model(cfg, key = subkey) # model_call(x_in, timesteps, parameters)
    #os.mkdir(wandb.run.dir)
      
    # Get optimizer and its parameters
    optimizer, optim_parameters = get_optim(cfg, model_parameters)

    # get loss functions and convert to grad function
    loss_fn = get_loss(cfg) 

    grad_fn = jax.jit(jax.grad(loss_fn,1)) 

    # start training for each epoch
    for epoch in range(cfg.train_and_test.train.epochs): 
        for i, (data, labels) in enumerate(train_dataset): # batch training

            # split key to keep randomness "random" for each training batch
            key, *subkey = jax.random.split(key, 4)


            # get grad for this batch
              # loss_value, grads = jax.value_and_grad(loss_fn)(model_parameters, model_call, data, labels, t) # is this extra computation time
            grads = grad_fn(model_call, model_parameters, data, labels)

            # get change in model_params and new optimizer params
              # optim_parameters, model_parameters = optim_alg(optim_parameters, model_parameters, t_data, labels)
            updates, optim_parameters = optimizer.update(grads, optim_parameters, model_parameters)

            # update model params
            model_parameters = optax.apply_updates(model_parameters, updates)

            # Logging loss and an image
            if i % cfg.wandb.log.frequency == 0:
                  if cfg.wandb.log.loss:
                    wandb.log({"loss": loss_fn(model_call, model_parameters, data, labels)})

                  if cfg.wandb.log.parameters:
                          with open(os.path.join(wandb.run.dir, "paremeters.pickle"), 'wb') as f:
                            pickle.dump((epoch*len(train_dataset) + i, model_parameters, optim_parameters), f, pickle.HIGHEST_PROTOCOL)
                          wandb.save("paramters.pickle")
        # Test loop
        if epoch % cfg.wandb.log.epoch_frequency == 0:
            pass
if __name__ == "__main__":
    run_experiment()
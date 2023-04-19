## Adding path (for some it is needed to import packages)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Saving and loading
import pickle

# convret to torch for FID
import torch

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
from optimizer.optimizers import get_optim

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

    # get sde
    SDE = get_sde(cfg)

    # get loss functions and convert to grad function
    loss_fn = get_loss(cfg) # loss_fn(func, function_parameters, data, perturbed_data, time, key)

    grad_fn = jax.grad(loss_fn,1) # TODO: try to JIT function partial(jax.jit,static_argnums=0)(jax.grad(loss_fn,1))

    # start training for each epoch
    for epoch in range(cfg.train_and_test.train.epochs): 
        for i, (data, labels) in enumerate(train_dataset): # batch training

            data = jax.device_put(data, sharding.reshape(n, 1))
            # split key to keep randomness "random" for each training batch
            key, *subkey = jax.random.split(key, 4)

            # get timesteps given random key for this batch and data shape
            timesteps = jax.random.uniform(subkey[0], (data.shape[0],), minval=0, maxval=1)

            # Perturb the data with the timesteps trhough sampling sde trick (for speed, see paper for explanation)
            perturbed_data = SDE.sample(timesteps, data, subkey[1])

            # scale timesteps for more significance
            scaled_timesteps = timesteps*999

            # get grad for this batch
              # loss_value, grads = jax.value_and_grad(loss_fn)(model_parameters, model_call, data, labels, t) # is this extra computation time
            grads = grad_fn(model_call, model_parameters, data, perturbed_data, scaled_timesteps, subkey[2])

            # get change in model_params and new optimizer params
              # optim_parameters, model_parameters = optim_alg(optim_parameters, model_parameters, t_data, labels)
            updates, optim_parameters = optimizer.update(grads, optim_parameters, model_parameters)

            # update model params
            model_parameters = optax.apply_updates(model_parameters, updates)

            # Logging loss and an image
            if i % cfg.wandb.log.frequency == 0:
                  if cfg.wandb.log.loss:
                    wandb.log({"loss": loss_fn(model_call, model_parameters, data, perturbed_data, scaled_timesteps, subkey[2])})
                    # wandb.log({"loss": loss_value})
                  if cfg.wandb.log.img:
                     # dt0 = - 1/N
                    drift = lambda t,y, args: SDE.reverse_drift(y, jnp.array([t]), args)
                    diffusion = lambda t,y, args: SDE.reverse_diffusion(y, jnp.array([t]), args)
                    get_sample = lambda t, key1, key0, xt: sample(0, 0, t.astype(float)[0], -1/1000, drift, diffusion, [model_call, model_parameters, key0], xt, key1) 
                    key, *subkey = jax.random.split(key, len(perturbed_data)*2 + 1)

                    args = (timesteps.reshape(-1,1), jnp.array(subkey[:len(subkey)//2]), jnp.array(subkey[len(subkey)//2:]), perturbed_data)

                    images = jax.vmap(get_sample, (0, 0, 0, 0))(*args)

                    # Rescale images for plotting
                    mins, maxs=jnp.min(perturbed_data, axis=1).reshape(-1, 1), jnp.max(perturbed_data, axis=1).reshape(-1,1)
                    rescaled_images = (perturbed_data-mins)/(maxs-mins)*255
                    display_images(cfg, images, labels)
                    display_images(cfg, perturbed_data, labels, log_title="Perturbed images")
                    display_images(cfg, rescaled_images, labels, log_title="Min-Max Rescaled")
                  if cfg.wandb.log.parameters:
                          with open(os.path.join(wandb.run.dir, "paremeters.pickle"), 'wb') as f:
                            pickle.dump((epoch*len(train_dataset) + i, model_parameters, optim_parameters), f, pickle.HIGHEST_PROTOCOL)
                          wandb.save("paramters.pickle")
        # Test loop
        if epoch % cfg.wandb.log.epoch_frequency == 0:
            pass
if __name__ == "__main__":
    run_experiment()
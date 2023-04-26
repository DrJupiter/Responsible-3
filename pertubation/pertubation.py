import jax.numpy as jnp
import jax.random as jrandom
import wandb

def get_pertubation(cfg):
    if cfg.pertubation.name == "laplace":
        if cfg.train_and_test.task == "classification":
            b = 2/cfg.pertubation.epsilon if cfg.pertubation.epsilon != 0 else 0
        elif cfg.train_and_test.task == "regression":
            b = 9/cfg.pertubation.epsilon if cfg.pertubation.epsilon != 0 else 0

        wandb.log({"epsilon": cfg.pertubation.epsilon})
        wandb.log({"pertubation scale": b})

        def lablace_sample(key, data, dtype=jnp.float32): 
            return data + jrandom.laplace(key, data.shape, dtype=dtype) * b 
    
        return lablace_sample
    raise NotImplementedError(f"Pertubation with name {cfg.pertubation.name} not found")
import jax.numpy as jnp
import jax.random as jrandom

def get_pertubation(cfg):
    if cfg.pertubation.name == "laplace":
        b = cfg.pertubation.scale
        def lablace_sample(key, data, dtype=jnp.float32): 
            return data + jrandom.laplace(key, data.shape, dtype=dtype) * b 
    
        return lablace_sample
    raise NotImplementedError(f"Pertubation with name {cfg.pertubation.name} not found")
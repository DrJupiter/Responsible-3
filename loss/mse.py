import jax.numpy as jnp

def mse(model_call, model_parameters, data, targets):
    predictions = model_call(model_parameters, data)
    diff = predictions-targets
    return 0.5 * jnp.mean(jnp.einsum("bc,bc->b", diff, diff)) # maybe add normalizing term for the weights
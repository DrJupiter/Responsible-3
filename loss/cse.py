import jax.numpy as jnp


def cross_entropy(model_call, model_parameters, data, targets):
  log_probs = model_call(model_parameters, data)
  target_class = jnp.argmax(targets, axis=1)
  nll = jnp.take_along_axis(log_probs, jnp.expand_dims(target_class, axis=1), axis=1)
  ce = -jnp.mean(nll)
  return ce

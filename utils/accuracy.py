import jax.numpy as jnp

def get_accuracy(cfg):
    if cfg.train_and_test.task == "classification":
        return classification_accuracy
    elif cfg.train_and_test.task == "regression":
        return lambda model_call, model_parameters, data, targets: 2 * mse(model_call, model_parameters, data, targets)
    raise NotImplementedError(f"Unable to find accuracy measure for {cfg.train_and_test.task}")


def classification_accuracy(model_call, params, data, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(model_call(params, data), axis=1)
  return jnp.mean(predicted_class == target_class)

from loss.mse import mse
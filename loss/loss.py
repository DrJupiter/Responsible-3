from loss.cse import cross_entropy
from loss.mse import mse


def get_loss(cfg):
    if cfg.loss.name == "cross_entropy":
        return cross_entropy
    elif cfg.loss.name == "mse":
        return mse
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
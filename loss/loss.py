
def get_loss(cfg):
    if cfg.loss.name == "cross_entropy":
        raise NotImplementedError("Implement")
    elif cfg.loss.name == "mse":
        return NotImplementedError("Implement")
    raise NotImplementedError(f"The loss {cfg.loss.name} is not implemented")
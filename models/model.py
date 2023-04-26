from models.classification import classification
from models.regression import regression

def get_model(cfg, key):
    if cfg.model.name == "classifier":
        return classification(cfg, key)
    elif cfg.model.name == "regression":
        return regression(cfg, key)
    raise NotImplementedError(f"Implement {cfg.model.name}")
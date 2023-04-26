import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torch import flatten as t_flatten
from torchvision.datasets import MNIST, CIFAR10
import multiprocessing as mp

import jax.numpy as jnp

def numpy_collate(batch):
  """
  Collation function for getting samples
  from `NumpyLoader`
  """
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  """
  The dataloader used for our image datasets
  """
  # TODO: Work with data in torch tensor until we pass it to model. As this loader is made to work on those, and is slow and has problems if not done like this.
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
      
class FlattenAndCast(object):
  """
  Flattens an image and converts the datatype to be 
  jax numpy's float32 type
  """
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

def dataload(cfg):
    """ 
    Returns the train and test dataset specified in the config
    in the form of a pytorch dataloader
    """

    transform = transforms.Compose([FlattenAndCast()])
    mnist_dataset_train = MNIST(cfg.dataset.path, download=True, transform=transform)
    training_generator = NumpyLoader(mnist_dataset_train, batch_size=cfg.train_and_test.train.batch_size, shuffle=cfg.train_and_test.train.shuffle) # num_workers=mp.cpu_count()

    mnist_dataset_test = MNIST(cfg.dataset.path, train=False, download=True, transform=transform)
    test_generator = NumpyLoader(mnist_dataset_test, batch_size=cfg.train_and_test.test.batch_size, shuffle=cfg.train_and_test.test.shuffle) # num_workers=mp.cpu_count()

    return training_generator, test_generator 

def label_transformation(cfg):
    if cfg.train_and_test.task == "classification":
      k = len(cfg.dataset.classes)
      def one_hot(x, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)
    if cfg.train_and_test.task == "regression":
        return lambda x: x
    
    raise NotImplementedError(f"Label transformation for {cfg.train_and_test.task} not found")

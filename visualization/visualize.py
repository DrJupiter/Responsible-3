import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import wandb
import jax
from torchvision.utils import make_grid
import numpy as np

def setup_plot():
    mpl.rcParams['lines.linewidth'] = 1
    #mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
    mpl.rcParams['font.family'] = 'Arial'

    
    #these don't work for some reason
    #mpl.rcParams['axes.titleweight'] = 'bold'
    #mpl.rcParams['axes.titlesize'] = '90'
    
    sns.set_theme(style="white", palette='pastel', font = 'Arial', font_scale=3)

    #sns.set_theme(style="white", palette='pastel', font = 'Microsoft Sans Serif', font_scale=1)
    #myFmt = mdates.DateFormatter('%b #Y')
    
    print("Plot settings applied")


def display_images(cfg, images, titles = [], rows = None, columns = 2, figsize= (7,7), pad=0.2, log_title=None):
    """
    Takes a list of images and plots them

    Takes the config, so we know how to plot the image in accordance with the dataset
    """
    
    if rows is None:
        rows = len(images)

    fig = plt.figure(figsize=figsize)

    # Title correction
    if isinstance(titles, np.ndarray):
        if cfg.dataset.name == 'cifar10':
            if titles.dtype == np.int64:
                titles = [cfg.dataset.classes[int(idx)] for idx in titles]


    for idx, img in enumerate(images):
        fig.add_subplot(rows, columns, idx+1) 
        if cfg.dataset.name == 'mnist':
            plt.imshow(img.reshape(28,28), cmap='gray')
        elif cfg.dataset.name == 'cifar10':
            plt.imshow((img).reshape(32,32,3)/255)
        plt.axis('off')
        if len(titles) == len(images):
            plt.title(titles[idx])
        else:
            plt.title(str(idx+1))
    plt.tight_layout(pad=pad) 

    if cfg.wandb.log.img:
        if wandb.run is None:
            run = wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)
        if log_title is not None:
            wandb.log({log_title: fig})
        else:
            wandb.log({f"plot {cfg.dataset.name}": fig})
    if cfg.visualization.visualize_img:
        plt.show()
    plt.close()


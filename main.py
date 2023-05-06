import os
import shutil

from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import setup_seed
from lib.loss import MetricLoss
from lib.tester import get_trainer
from configs.models import architectures
from configs.config import Config
from torch import optim


def backup(option=False):
    if option:
        os.system(f'cp -r models {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py', config.snapshot_dir)


setup_seed(0)

if __name__ == '__main__':
    # load configurations
    config = Config()

    # backup the current file options and codes True or False
    backup()

    # model initialization
    config.architecture = architectures[config.dataset]
    config.model = KPFCNN(config)

    # create optimizer
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(config.model.parameters(),
                                     lr=config.lr,
                                     momentum=config.momentum,
                                     weight_decay=config.weight_decay,
                                     )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(config.model.parameters(),
                                      lr=config.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=config.weight_decay,
                                      )

    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(config.optimizer,
                                                        gamma=config.scheduler_gamma,
                                                        )

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                                              batch_size=config.batch_size,
                                                              shuffle=True,
                                                              num_workers=config.num_workers,
                                                              )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=1,
                                          neighborhood_limits=neighborhood_limits
                                          )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=1,
                                           neighborhood_limits=neighborhood_limits)

    # create evaluation metrics
    config.desc_loss = MetricLoss(config)
    trainer = get_trainer(config)
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'val':
        trainer.eval()
    else:
        trainer.test()
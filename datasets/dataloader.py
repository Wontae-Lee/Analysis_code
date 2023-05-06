

from datasets.indoor import IndoorDataset

def get_datasets(config):
    if (config.dataset == 'indoor'):
        info_train = load_obj(config.train_info)
        info_val = load_obj(config.val_info)
        info_benchmark = load_obj(f'configs/indoor/{config.benchmark}.pkl')

        train_set = IndoorDataset(info_train, config, data_augmentation=True)
        val_set = IndoorDataset(info_val, config, data_augmentation=False)
        benchmark_set = IndoorDataset(info_benchmark, config, data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set


def get_dataloader(dataset, batch_size=1, num_workers=4, shuffle=True, neighborhood_limits=None):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
    print("neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor, config=dataset.config, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader, neighborhood_limits


if __name__ == '__main__':
    pass
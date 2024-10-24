import torch
import pickle
import numpy as np

from torch.utils.data import DataLoader

# Select the window of tactile data
def window_select(log, path, f ,idx, window):
    if window == 0:
        d = pickle.load(open(path + '/' + str(idx) + '.p', "rb"))

        return np.reshape(d[0], (1,96,96)), d[1], d[2], np.reshape(d[0], (1,96,96))

    max_len = log[f+1]
    min_len = log[f]
    l = max([min_len, idx-window])
    u = min([max_len, idx+window])

    dh = pickle.load(open(path + '/' + str(idx) + '.p', "rb"))
    heatmap = dh[1]
    keypoint = dh[2]

    tactile = np.empty((2*window, 96, 96))

    if l == min_len:
        for i in range(min_len, min_len+2*window):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i-min_len,:,:] = d[0]

        return tactile, heatmap, keypoint

    elif u == max_len:
        for i in range(max_len-2*window, max_len):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i-(max_len-2*window),:,:] = d[0]

        return tactile, heatmap, keypoint

    else:
        for i in range(l, u):
            d = pickle.load(open(path + '/' + str(i) + '.p', "rb"))
            tactile[i-l,:,:] = d[0]

        return tactile, heatmap, keypoint

# Create DDP sampler
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers

# Create dataloader
def create_dataloader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
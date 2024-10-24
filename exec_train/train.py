import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import yaml
import torch
import datetime
import torch.distributed as dist

from pathlib import Path

from utils.system import get_configs
from utils.display import print_header
from exec_train.utils_model import set_seed
from exec_train.utils_learn import plot_learn
from exec_train.utils_dist import init_distributed_mode, is_main_process
from exec_train.utils_train import setup_data, setup_model, train, eval, save_log, early_stop_check

# Main
def main(configs):
    # Initialize multi-gpu distributed process
    if configs['distributed']:
        print_header("Initialize Distributed Mode")
        init_distributed_mode(configs)

    # Setup model
    model, model_without_ddp, optimizer, scheduler, start_epoch = setup_model(configs)

    # Setup data
    train_dataloader, val_dataloader = setup_data(configs)

    # Loss collectors
    loss_collect = {'train': {}, 'val': {}}

    # Early stop params
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Start training
    print_header("Start Training")
    start_time = time.time()
    for epoch in range(start_epoch + 1, configs['max_epoch'] + 1):
        # Print
        print_header(f"Epoch {epoch}")

        # Train model
        train_stats = train(epoch, model, train_dataloader, optimizer, configs)

        # Check main process (rank == 0) (automatically goes into for not multi-gpu)
        if is_main_process():
            # Eval model
            val_stats = eval(epoch, model_without_ddp, val_dataloader, configs)

            # Collect losses
            [loss_collect[phase].setdefault(key, []).append(locals()[f"{phase}_stats"][key]) for key in val_stats for phase in ['train', 'val']]

            # Save and log
            save_log(model_without_ddp, optimizer, configs, epoch, train_stats, val_stats)


            # Step scheduler
            scheduler.step(val_stats['total_loss'])

            # Early stop check
            early_stop, best_loss, epochs_without_improvement = early_stop_check(val_stats, model_without_ddp, optimizer, configs, epoch, best_loss, epochs_without_improvement)
            if early_stop:
                break

        # Synchronize multi-gpu
        if configs['distributed']:
            dist.barrier()
            torch.cuda.empty_cache()

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return loss_collect

if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Get configs
    configs = yaml.load(open(get_configs() / 'train' / 'train.yaml', 'r'), Loader=yaml.Loader)
    Path(configs['output_dir']).mkdir(parents=True, exist_ok=True); yaml.dump(configs, open(os.path.join(configs['output_dir'], 'configs.yaml'), 'w'))

    # Execute main
    loss_data = main(configs)

    # Plot learning curves
    plot_learn(loss_data, configs['output_dir'])
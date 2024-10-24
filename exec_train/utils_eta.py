import math

# Cosine learning rate schedule
def cosine_lr_schedule(optimizer, group, epoch, max_epoch, init_lr, min_lr):
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    optimizer.param_groups[group]['lr'] = lr

# Warmup learning rate schedule
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Step learning rate schedule
def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
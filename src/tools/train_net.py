#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import wandb
import torch

from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.optimizer as optim
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.models.losses as losses
import slowfast.utils.distributed as du
import slowfast.utils.checkpoint as cu
import slowfast.utils.misc as misc

from slowfast.utils.mixup import mixup_data, mixup_criterion
from slowfast.utils.meters import TrainMeter, ValMeter
from slowfast.models import build_model
from slowfast.datasets import loader

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, wandb_log=False
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    data_size = len(train_loader)

    if cfg.NUM_GPUS > 1:
        model.module.freeze_fn('bn_statistics')
    else:
        model.freeze_fn('bn_statistics')

    train_meter.iter_tic()
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if hasattr(val[i], 'cuda'):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    if hasattr(val, 'cuda'):
                        meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=cfg.MIXUP.ALPHA)
        preds = model(inputs)

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction=cfg.MODEL.LOSS_REDUCTION)

        # Compute the loss.
        if cfg.MIXUP.ENABLE:
            loss = mixup_criterion(
                    loss_fun, 
                    preds, 
                    labels_a, 
                    labels_b, 
                    lam
                )
        else:
            loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce(
                [loss, top1_err, top5_err]
            )
            meta = du.all_gather_unaligned(meta)
            metadata = {'annotation_id': []}
            for i in range(len(meta)):
                metadata['annotation_id'].extend(meta[i]['annotation_id'])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )

        if cfg.NUM_GPUS > 1:
            preds, labels = du.all_gather([preds, labels])

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            preds.detach(),
            labels.detach(),
            [pg['lr'] for pg in optimizer.param_groups],
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            )  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        if wandb_log:
            wandb.log(
                {
                    "Train/loss": loss,
                    "Train/lr": lr[0] if isinstance(lr, (list,)) else lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                    "train_step": data_size * cur_epoch + cur_iter,
                },
            )
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    scores_dict = train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    if wandb_log:
        wandb.log(
            {
                "Train/epoch/Top1_err": scores_dict["top1_err"], 
                "Train/epoch/Top5_err": scores_dict["top5_err"],
                "Train/epoch/mAP": scores_dict["mAP"],
                "Train/epoch/mAUC": scores_dict["mAUC"],
                "Train/epoch/mPCA": scores_dict["mPCA"],
                "epoch": cur_epoch
            }
        )

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_log=False):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    data_size = len(val_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if hasattr(val[i], 'cuda'):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    if hasattr(val, 'cuda'):
                        meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()


        preds = model(inputs)

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce(
                [loss, top1_err, top5_err]
            )

        # Copy the errors from GPU to CPU (sync point).
        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )
                
        if cfg.NUM_GPUS > 1:
            preds, labels = du.all_gather([preds, labels])
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            preds,
            labels,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ) # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        val_step = (cur_epoch // cfg.TRAIN.EVAL_PERIOD) * data_size 
        if wandb_log:
            wandb.log(
                {
                    "Val/loss": loss,
                    "Val/Top1_err": top1_err,
                    "Val/Top5_err": top5_err,
                    "val_step": val_step + cur_iter,
                },
            )
        val_meter.iter_toc()
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch, scores_dict = val_meter.log_epoch_stats(cur_epoch)

    if wandb_log:
        wandb.log(
            {
                "Val/epoch/Top1_err": scores_dict["top1_err"], 
                "Val/epoch/Top5_err": scores_dict["top5_err"],
                "Val/epoch/mAP": scores_dict["mAP"],
                "Val/epoch/mAUC": scores_dict["mAUC"],
                "Val/epoch/mPCA": scores_dict["mPCA"],
                "epoch": cur_epoch
            }
        )

    top1 = scores_dict["top1_err"]
    val_meter.reset()
    return is_best_epoch, top1

def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (dict,)):
                    for k, v in inputs.items():
                        for i in range(len(v)):
                            inputs[k][i] = v[i].cuda(non_blocking=True)
                elif isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.BN.FREEZE:
        if cfg.NUM_GPUS > 1:
            model.module.freeze_fn('bn_parameters')
        else:
            model.freeze_fn('bn_parameters')
        
    if cfg.MODEL.FREEZE_BACKBONE:
        if cfg.NUM_GPUS > 1:
            model.module.freeze_fn('freeze_backbone')
        else:
            model.freeze_fn('freeze_backbone')

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train")
        if cfg.BN.USE_PRECISE_STATS else None
    )

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    if cfg.WANDB.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        wandb_log = True
        if cfg.TRAIN.AUTO_RESUME and cfg.WANDB.RUN_ID != "":
            wandb.init(project=cfg.MODEL.MODEL_NAME, config=cfg, sync_tensorboard=True, resume=cfg.WANDB.RUN_ID)
        else:
            wandb.init(project=cfg.MODEL.MODEL_NAME, config=cfg, sync_tensorboard=True)
        wandb.watch(model)

    else:
        wandb_log = False

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, wandb_log
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            is_best_epoch, _ = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_log)
            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=is_best_epoch)
    if wandb_log:
        wandb.finish()
    du.synchronize()
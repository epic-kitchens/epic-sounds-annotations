#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import pickle
import torch
import os

from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging
import slowfast.utils.distributed as du
import slowfast.utils.checkpoint as cu
import slowfast.utils.misc as misc

from slowfast.utils.meters import TestMeter
from slowfast.models import build_model
from slowfast.datasets import loader

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis.. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        if hasattr(val[i], 'cuda'):
                            val[i] = val[i].cuda(non_blocking=True)
                else:
                    if hasattr(val, 'cuda'):
                        meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()
        # Perform the forward pass.
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather(
                [preds, labels, video_idx]
            )
            meta = du.all_gather_unaligned(meta)
            metadata = {'annotation_id': []}
            for i in range(len(meta)):
                metadata['annotation_id'].extend(meta[i]['annotation_id'])
            meta = metadata
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            preds.detach(), labels.detach(), video_idx.detach(), meta
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()

    if cfg.TEST.SAVE_RESULTS_PATH != "":
        save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

        with PathManager.open(save_path, "wb") as f:
            pickle.dump([all_preds, all_labels], f)

        logger.info(
            "Successfully saved prediction results to {}".format(save_path)
        )

    preds, preds_clips, labels, meta = test_meter.finalize_metrics()
    return test_meter, preds, preds_clips, labels, meta

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
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
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(
                            model, 
                            cfg, 
                            use_train_input=False
                        )

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    test_meter = TestMeter(
        len(test_loader.dataset) // (cfg.TEST.NUM_ENSEMBLE_VIEWS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        cfg.MODEL.NUM_CLASSES[0],
        len(test_loader),
        cfg.EPICSOUNDS.TEST_LIST
    )

    # Perform multi-view test on the entire dataset.
    _, preds, _, _, annotation_ids = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        results = {'annotation_id': annotation_ids, 'interaction_output': preds}
        scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
        if not os.path.exists(scores_path):
            os.makedirs(scores_path)
        file_path = os.path.join(scores_path, cfg.EPICSOUNDS.TEST_LIST)
        pickle.dump(results, open(file_path, 'wb'))

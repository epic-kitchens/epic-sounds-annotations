#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
from fvcore.common.timer import Timer

import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc

logger = logging.get_logger(__name__)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count

class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.iter_times = []
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.iter_times = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()
        self.iter_times.append(self.iter_timer.seconds())

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, preds, labels, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        avg_iter_length = sum(self.iter_times) / len(self.iter_times)
        eta_sec = avg_iter_length * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        stats["top1_err"] = self.mb_top1_err.get_win_median()
        stats["top5_err"] = self.mb_top5_err.get_win_median()
                
        mini_stats = metrics.get_stats(
            torch.cat(self.all_preds).cpu().numpy(),
            torch.cat(self.all_labels).cpu().numpy()
        )
        stats.update(mini_stats)
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        avg_iter_length = sum(self.iter_times) / len(self.iter_times)
        eta_sec = avg_iter_length * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats["top1_err"] = top1_err
        stats["top5_err"] = top5_err
        stats["loss"] = avg_loss
               
        mini_stats = metrics.get_stats(
            torch.cat(self.all_preds).cpu().numpy(),
            torch.cat(self.all_labels).cpu().numpy()            
        )
        stats.update(mini_stats)
        logging.log_json_stats(stats)

        scores_dict = {
                "top1_err": top1_err, 
                "top5_err": top5_err,
                "mAP": stats["mAP"],
                "mAUC": stats["mAUC"],
                "mPCA": stats["mPCA"]
            }
            
        return scores_dict

class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.iter_times = []
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        self.max_map = 0.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.iter_times = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()
        self.iter_times.append(self.iter_timer.seconds())

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, preds, labels, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
    
        self.all_preds.append(preds)
        self.all_labels.append(labels)
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        avg_iter_length = sum(self.iter_times) / len(self.iter_times)
        eta_sec = avg_iter_length * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        
        mini_stats = metrics.get_stats(
            torch.cat(self.all_preds).cpu().numpy(),
            torch.cat(self.all_labels).cpu().numpy()
        )
        stats.update(mini_stats)
        stats["top1_err"] = self.mb_top1_err.get_win_median()
        stats["top5_err"] = self.mb_top5_err.get_win_median()

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        
        mini_stats = metrics.get_stats(
            torch.cat(self.all_preds).cpu().numpy(),
            torch.cat(self.all_labels).cpu().numpy(),
        )
        stats.update(mini_stats)

        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        is_best_epoch = top1_err < self.min_top1_err
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)

        stats["top1_err"] = top1_err
        stats["top5_err"] = top5_err
        stats["min_top1_err"] = self.min_top1_err
        stats["min_top5_err"] = self.min_top5_err

        logging.log_json_stats(stats)

        scores_dict = {
                "top1_err": top1_err, 
                "top5_err": top5_err,
                "mAP": stats["mAP"],
                "mAUC": stats["mAUC"],
                "mPCA": stats["mPCA"]
            }

        return is_best_epoch, scores_dict

class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        split
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.iter_times = []
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.split = split
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_preds_clips = torch.zeros((num_videos, num_clips, num_cls))
        self.video_labels = torch.zeros((num_videos)).long()
        self.clip_count = torch.zeros((num_videos)).long()
        self.metadata = np.zeros(num_videos, dtype=object)
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_preds_clips.zero_()
        self.video_labels.zero_()
        self.iter_times = []
        self.metadata.fill(0)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()
        self.iter_times.append(self.iter_timer.seconds())

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, preds, labels, clip_ids, metadata):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            clip_temporal_id = int(clip_ids[ind]) % self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            self.video_preds[vid_id] += preds[ind]
            self.video_preds_clips[vid_id, clip_temporal_id] = preds[ind]
            self.metadata[vid_id] = metadata['annotation_id'][ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        avg_iter_length = sum(self.iter_times) / len(self.iter_times)        
        eta_sec = avg_iter_length * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            non_match = torch.where(self.clip_count != self.num_clips)[0]
            non_match_count = self.clip_count[non_match]
            mismatch = torch.stack([non_match, non_match_count], dim=1)
            logger.warning(
                "clip count [{}] ~= num clips {}".format(
                    ", ".join(["{}: {}".format(i, k) for i, k in mismatch]),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}

        if not "test" in self.split:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
            
            mini_stats = metrics.get_stats(
                self.video_preds.cpu().numpy(),
                self.video_labels.cpu().numpy()
            )
            self.stats.update(mini_stats)

        logging.log_json_stats(self.stats)
        return self.video_preds.numpy().copy(), \
               self.video_preds_clips.numpy().copy(), \
               F.one_hot(self.video_labels, num_classes=self.video_preds.shape[1]).numpy().copy(), \
               self.metadata.copy()

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import uuid
import pathlib
from typing import List, Optional

import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader
import data.utils as utils

from .util import get_opts, move_to
from .callbacks import Callback, ConsoleLogger, Checkpoint, CheckpointSaver, TensorboardLogger


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result


def _div_dict(d, n):
    result = dict(d)
    for k in result:
        result[k] /= n
    return result


class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """
    def __init__(
            self,
            game: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_data: DataLoader,
            validation_data: Optional[DataLoader] = None,
            device: torch.device = None,
            callbacks: Optional[List[Callback]] = None
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.game.to(self.device)
        # NB: some optimizers pre-allocate buffers before actually doing any steps
        # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
        # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
        self.optimizer.state = move_to(self.optimizer.state, self.device)
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks

        if common_opts.load_from_checkpoint is not None:
            print(f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}")
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        if common_opts.preemptable:
            assert common_opts.checkpoint_dir, 'checkpointing directory has to be specified'
            d = self._get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
            self.checkpoint_path = d
            self.load_from_latest(d)
        else:
            self.checkpoint_path = None if common_opts.checkpoint_dir is None \
                else pathlib.Path(common_opts.checkpoint_dir)

        if self.checkpoint_path:
            checkpointer = CheckpointSaver(checkpoint_path=self.checkpoint_path, checkpoint_freq=common_opts.checkpoint_freq)
            self.callbacks.append(checkpointer)

        if common_opts.tensorboard:
            assert common_opts.tensorboard_dir, 'tensorboard directory has to be specified'
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]

    def _get_preemptive_checkpoint_dir(self, checkpoint_root):
        if 'SLURM_JOB_ID' not in os.environ:
            print('Preemption flag set, but I am not running under SLURM?')

        # start assuming the job runs as an array
        # fallback, if not
        job_id = os.environ.get('SLURM_ARRAY_JOB_ID', None)
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)

        if job_id is None or task_id is None:
            job_id = os.environ.get('SLURM_JOB_ID', uuid.uuid4())
            task_id = os.environ.get('SLURM_PROCID', 0)

        d = pathlib.Path(checkpoint_root) / f'{job_id}_{task_id}'
        d.mkdir(exist_ok=True)

        return d

    def eval(self, return_acc=False):
        mean_loss = 0.0
        mean_rest = {}
        accuracy = 0

        n_batches = 0
        self.game.eval()
        with torch.no_grad():
            for batch in self.validation_data:
                reference = batch[2]
                distractors = batch[3]
                batch = batch[:2]

                batch = move_to(batch, self.device)
                reference = move_to(reference, self.device)
                distractors = move_to(distractors, self.device)

                message = self.game.sender((batch[0], reference))
                receiver_output = self.game.receiver(message, reference)[:,-1,:]

                target_repr = self.game.receiver.agent.vision(batch[0])
                target_MSE = torch.mean(F.mse_loss(receiver_output, target_repr, reduction="none"),1)
                correct = torch.ones(distractors.size(0)).type(torch.FloatTensor)

                for i in range(distractors.size(1)):
                  distract_repr = self.game.receiver.agent.vision(distractors[:,i,:,:,:])
                  distract_MSE = torch.mean(F.mse_loss(receiver_output, distract_repr,reduction="none"),1)
                  correct[distract_MSE < target_MSE] = 0
                accuracy += (torch.sum(correct) / distractors.size(0))

                optimized_loss, rest = self.game(*batch, reference, distractors)

                mean_loss += optimized_loss
                mean_rest = _add_dicts(mean_rest, rest)
                n_batches += 1
        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        accuracy /= n_batches
        print(f"{{ \"test_accuracy\": {accuracy.item()}, \"distractors\": {distractors.size(1)} }}")
        if return_acc:
          return mean_loss.item(), mean_rest, accuracy
        return mean_loss.item(), mean_rest

    def train_epoch(self):
        mean_loss = 0
        mean_rest = {}
        n_batches = 0
       
        self.game.train()
        for i, batch in enumerate(self.train_data):
            if self.train_data.dataset.dynamic:
              reference = utils.reconstruct("square", utils.alter_attributes("square", batch[1]))
              distractors = batch[3]
              batch = batch[:2]
            else:
              distractors = batch[3]
              reference = batch[2]
              batch = batch[:2]
              
            self.optimizer.zero_grad()
            batch = move_to(batch, self.device)
            reference = move_to(reference, self.device)
            distractors = move_to(distractors, self.device)

            optimized_loss, rest = self.game(*batch, reference, distractors)
            mean_rest = _add_dicts(mean_rest, rest)
            optimized_loss.backward()
            self.optimizer.step()

            n_batches += 1
            mean_loss += optimized_loss

        mean_loss /= n_batches
        mean_rest = _div_dict(mean_rest, n_batches)
        return mean_loss.item(), mean_rest

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin()

            train_loss, train_rest = self.train_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_rest)

            if self.validation_data is not None and self.validation_freq > 0 and epoch % self.validation_freq == 0:
                for callback in self.callbacks:
                    callback.on_test_begin()
                validation_loss, rest = self.eval()
                for callback in self.callbacks:
                    callback.on_test_end(validation_loss, rest)

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f'# loading trainer state from {path}')
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob('*.tar'):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)

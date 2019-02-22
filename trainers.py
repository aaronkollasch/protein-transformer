import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from data_loaders import GeneratorDataLoader, IPITrainTestDataset
from model_logging import Logger
from functions import NoamOpt, make_std_mask


class TransformerTrainer:
    default_params = {
        'optimizer': 'Adam',
        'lr': 0.0,
        'lr_factor': 2,
        'lr_warmup': 4000,
        'weight_decay': 0,
        'clip': 10.0,
        'opt_params': {},
        'snapshot_path': None,
        'snapshot_name': 'snapshot',
        'snapshot_interval': 1000,
    }

    def __init__(
            self,
            model,
            data_loader,
            optimizer=None,
            params=None,
            lr=None,
            weight_decay=None,
            gradient_clipping=None,
            logger=Logger(),
            snapshot_path=None,
            snapshot_name=None,
            snapshot_interval=None,
            snapshot_exec_template=None,
            device=torch.device('cpu')
    ):
        self.params = self.default_params.copy()
        self.params.update(model.hyperparams['optimization'])
        if params is not None:
            self.params.update(params)
        if optimizer is not None:
            self.params['optimizer'] = optimizer
        if lr is not None:
            self.params['lr'] = lr
        if weight_decay is not None:
            self.params['weight_decay'] = weight_decay
        if gradient_clipping is not None:
            self.params['clip'] = gradient_clipping
        if snapshot_path is not None:
            self.params['snapshot_path'] = snapshot_path
        if snapshot_name is not None:
            self.params['snapshot_name'] = snapshot_name
        if snapshot_interval is not None:
            self.params['snapshot_interval'] = snapshot_interval
        if snapshot_exec_template is not None:
            self.params['snapshot_exec_template'] = snapshot_exec_template
        if self.params['weight_decay'] > 0:
            self.params['weight_decay'] = self.params['weight_decay'] / data_loader.dataset.n_eff

        self.model = model
        self.loader = data_loader

        self.run_fr = 'fr' in model.MODEL_TYPE
        self.optimizer_type = getattr(optim, self.params['optimizer'])
        self.logger = logger
        self.logger.trainer = self
        self.device = device

        self.optimizer = NoamOpt(
            model_size=self.model.hyperparams['transformer']['d_model'],
            factor=self.params['lr_factor'],
            warmup=self.params['lr_warmup'],
            optimizer=self.optimizer_type(
                params=self.model.parameters(),
                lr=self.params['lr'],
                weight_decay=self.params['weight_decay'],
                **self.params['opt_params'],
            ),
        )

    def train(self, steps=1e8):
        self.model.train()

        data_iter = iter(self.loader)
        n_eff = self.loader.dataset.n_eff

        # print('    step  step-t load-t   loss       CE-loss    bitperchar', flush=True)
        for step in range(int(self.model.step) + 1, int(steps) + 1):
            self.model.step = step
            # start = time.time()

            batch = next(data_iter)
            for key in batch.keys():
                batch[key] = batch[key].to(self.device, non_blocking=True)
            # data_load_time = time.time()-start

            try:
                if self.run_fr:
                    src_mask, tgt_mask = make_std_mask(batch['decoder_input'], batch['decoder_input'])
                    _, tgt_mask_r = make_std_mask(None, batch['decoder_input_r'])
                    output_logits_f, output_logits_r = self.model(
                        batch['decoder_input'], batch['decoder_input'],
                        src_mask, tgt_mask,
                        batch['decoder_input_r'], tgt_mask_r)
                    losses = self.model.calculate_loss(
                        output_logits_f, batch['decoder_output'], batch['decoder_mask'], n_eff,
                        output_logits_r, batch['decoder_output_r'], batch['decoder_mask'], n_eff
                    )
                else:
                    src_mask, tgt_mask = make_std_mask(batch['decoder_input'], batch['decoder_input'])
                    output_logits = self.model(batch['decoder_input'], batch['decoder_input'],
                                               src_mask, tgt_mask)
                    losses = self.model.calculate_loss(
                        output_logits, batch['decoder_output'], batch['decoder_mask'], n_eff=n_eff)

                if step in [10, 100, 1000, 10000]:
                    print(f'step {step:6d}: '
                          f'GPU Mem Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB, '
                          f'Cached: {round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)} GB',
                          flush=True)
            except RuntimeError as e:
                print("out of memory at ", step, "with input size", batch['decoder_input'].shape, flush=True)
                raise e

            self.optimizer.zero_grad()
            losses['loss'].backward()

            if self.params['clip'] is not None:
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
            else:
                total_norm = 0.0

            self.optimizer.step()

            for key in losses:
                losses[key] = losses[key].detach()
                if self.run_fr and 'per_seq' not in key and '_f' not in key and '_r' not in key:
                    losses[key] /= 2
            losses.update({'grad_norm': total_norm})

            if step % self.params['snapshot_interval'] == 0:
                if self.params['snapshot_path'] is not None:
                    self.save_state()

            self.logger.log(step, losses, total_norm)
            # print("{: 8d} {:6.3f} {:5.4f} {:11.6f} {:11.6f} {:11.8f}".format(
            #     step, time.time() - start, data_load_time,
            #     losses['loss'], losses['ce_loss'], losses['bitperchar']), flush=True)

    def validate(self):
        if not isinstance(self.loader.dataset, IPITrainTestDataset):
            return None
        self.model.eval()
        self.loader.dataset.test()
        self.loader.dataset.unlimited_epoch = False

        with torch.no_grad():
            loader = GeneratorDataLoader(self.loader.dataset, num_workers=self.loader.num_workers)
            pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)

            true_outputs = []
            logits = []
            losses = []
            accuracies = []
            for batch in loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device, non_blocking=True)

                log_probs, y_choices = self.model.predict_all_y(batch['input'], batch['mask'], batch['decoder_output'])
                output_logits = self.model.predict_logits(log_probs, y_choices)

                error = self.model.calculate_accuracy(
                    output_logits, batch['label'], reduction='none')
                ce_loss = F.binary_cross_entropy_with_logits(
                    output_logits, batch['label'], pos_weight=pos_weights, reduction='none')

                true_outputs.append(batch['label'])
                logits.append(output_logits)
                losses.append(ce_loss)
                accuracies.append(error)

            true_outputs = torch.cat(true_outputs, 0).cpu().numpy()
            logits = torch.cat(logits, 0).cpu().numpy()
            roc_scores = roc_auc_score(true_outputs, logits, average=None)
            if isinstance(roc_scores, np.ndarray):
                roc_scores = roc_scores.tolist()
            else:
                roc_scores = [roc_scores]

            true_outputs = true_outputs.mean(0).tolist()
            logits = logits.mean(0).tolist()
            losses = torch.cat(losses, 0).mean(0).tolist()
            accuracies = torch.cat(accuracies, 0).mean(0).tolist()

        self.model.train()
        self.loader.dataset.train()
        self.loader.dataset.unlimited_epoch = True
        return losses, accuracies, true_outputs, logits, roc_scores

    def test(self, data_loader, model_eval=True, num_samples=1):  # TODO implement
        if model_eval:
            self.model.eval()

        print('sample    step  step-t  CE-loss     bit-per-char', flush=True)
        output = {
            'name': [],
            'mean': [],
            'forward': [],
            'reverse': [],
            'bitperchar': [],
            'sequence': []
        }
        if not self.run_fr:
            del output['forward']
            del output['reverse']

        for i_iter in range(num_samples):
            output_i = {
                'name': [],
                'mean': [],
                'forward': [],
                'reverse': [],
                'bitperchar': [],
                'sequence': []
            }
            if not self.run_fr:
                del output['forward']
                del output['reverse']

            for i_batch, batch in enumerate(data_loader):
                start = time.time()
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.run_fr:
                        src_mask, tgt_mask = make_std_mask(batch['decoder_input'], batch['decoder_input'])
                        _, tgt_mask_r = make_std_mask(None, batch['decoder_input_r'])
                        output_logits_f, output_logits_r = self.model(
                            batch['decoder_input'], batch['decoder_input'],
                            src_mask, tgt_mask,
                            batch['decoder_input_r'], tgt_mask_r)
                        losses = self.model.reconstruction_loss(
                            output_logits_f, batch['decoder_output'], batch['decoder_mask'],
                            output_logits_r, batch['decoder_output_r'], batch['decoder_mask']
                        )
                    else:
                        src_mask, tgt_mask = make_std_mask(batch['decoder_input'], batch['decoder_input'])
                        output_logits = self.model(batch['decoder_input'], batch['decoder_input'],
                                                   src_mask, tgt_mask)
                        losses = self.model.reconstruction_loss(
                            output_logits, batch['decoder_output'], batch['decoder_mask'])

                    ce_loss_per_seq = losses['ce_loss_per_seq'].cpu()
                    bitperchar_per_seq = losses['bitperchar_per_seq'].cpu()

                    if self.run_fr:
                        ce_loss_per_seq = ce_loss_per_seq.mean(0)
                        bitperchar_per_seq = bitperchar_per_seq.mean(0)

                output_i['name'].extend(batch['names'])
                output_i['sequence'].extend(batch['sequences'])
                output_i['mean'].extend(ce_loss_per_seq.numpy())
                output_i['bitperchar'].extend(bitperchar_per_seq.numpy())

                print("{: 4d} {: 8d} {:6.3f} {:11.6f} {:11.6f}".format(
                    i_iter, i_batch, time.time()-start, ce_loss_per_seq.mean(), bitperchar_per_seq.mean()),
                    flush=True)

            output['name'] = output_i['name']
            output['sequence'] = output_i['sequence']
            output['bitperchar'].append(output_i['bitperchar'])
            output['mean'].append(output_i['mean'])

        output['bitperchar'] = np.array(output['bitperchar']).mean(0)
        output['mean'] = np.array(output['mean']).mean(0)

        self.model.train()
        return output

    def save_state(self, last_batch=None):
        snapshot = f"{self.params['snapshot_path']}/{self.params['snapshot_name']}_{self.model.step}.pth"
        revive_exec = f"{self.params['snapshot_path']}/revive_executable/{self.params['snapshot_name']}.sh"
        if not os.path.exists(os.path.dirname(snapshot)):
            os.makedirs(os.path.dirname(snapshot), exist_ok=True)
        if not os.path.exists(os.path.dirname(revive_exec)):
            os.makedirs(os.path.dirname(revive_exec), exist_ok=True)
        torch.save(
            {
                'step': self.model.step,
                'model_type': self.model.MODEL_TYPE,
                'model_state_dict': self.model.state_dict(),
                'model_dims': self.model.dims,
                'model_hyperparams': self.model.hyperparams,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_params': self.params,
                'dataset_params': self.loader.dataset.params,
                'last_batch': last_batch
            },
            snapshot
        )
        with open(revive_exec, "w") as f:
            snapshot_exec = self.params['snapshot_exec_template'].format(
                restore=os.path.abspath(snapshot)
            )
            f.write(snapshot_exec)

    def load_state(self, checkpoint, map_location=None):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        if self.model.MODEL_TYPE != checkpoint['model_type']:
            print("Warning: model type mismatch: loaded type {} for model type {}".format(
                checkpoint['model_type'], self.model.MODEL_TYPE
            ))
        if self.model.hyperparams != checkpoint['model_hyperparams']:
            print("Warning: model hyperparameter mismatch")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.step = checkpoint['step']
        self.params.update(checkpoint['train_params'])

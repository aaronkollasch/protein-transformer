#!/usr/bin/env python
import sys
import argparse
import time
import json
import warnings
import math

import numpy as np
import torch

import data_loaders
import models
import trainers
import model_logging
from utils import get_cuda_version, get_cudnn_version

working_dir = '/n/groups/marks/users/aaron/transformer'
data_dir = '/n/groups/marks/projects/antibodies'


###################
# PARSE ARGUMENTS #
###################

parser = argparse.ArgumentParser(description="Train a transformer model on a collection of sequences.")
parser.add_argument("--model-type", type=str, default='transformer',
                    help="Choose model type")
parser.add_argument("--preset", type=str, default=None,
                    help="Choose hyperparameter preset")
parser.add_argument("--d-model", metavar='D', type=int, default=512,
                    help="Number of channels in attention head.")
parser.add_argument("--d-ff", metavar='D', type=int, default=2048,
                    help="Number of channels in attention head.")
parser.add_argument("--num-heads", type=int, default=8,
                    help="Number of attention heads.")
parser.add_argument("--num-layers", type=int, default=6,
                    help="Number of layers.")
parser.add_argument("--batch-size", metavar='N', type=int, default=10,
                    help="Batch size.")
parser.add_argument("--num-iterations", type=int, default=250005,
                    help="Number of iterations to run the model.")
parser.add_argument("--num-epochs", type=int, default=None,
                    help="Number of epochs to run the model. Disables num-iterations.")
parser.add_argument("--dataset",  metavar='D', type=str, default=None,
                    help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
parser.add_argument("--num-data-workers", metavar='N', type=int, default=8,
                    help="Number of workers to load data")
parser.add_argument("--restore", type=str, default=None,
                    help="Snapshot path for restoring a model to continue training.")
parser.add_argument("--run-name", metavar='NAME', type=str, default=None,
                    help="Name of run")
parser.add_argument("--r-seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--dropout-p", type=float, default=0.1,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU training")
args = parser.parse_args()


########################
# MAKE RUN DESCRIPTORS #
########################

if 'small' in args.preset.split('-'):
    args.d_model = 128
    args.d_ff = 512
    args.num_heads = 4
    args.num_layers = 6
elif 'medium' in args.preset.split('-'):
    args.d_model = 256
    args.d_ff = 1024
    args.num_heads = 8
    args.num_layers = 6
elif 'large' in args.preset.split('-'):
    args.d_model = 512
    args.d_ff = 2048
    args.num_heads = 8
    args.num_layers = 6
elif 'XS' in args.preset.split('-'):
    args.d_model = 64
    args.d_ff = 256
    args.num_heads = 4
    args.num_layers = 6
elif args.preset is not None:
    warnings.warn(f"Unrecognized preset: {args.preset}")

if args.run_name is None:
    if args.preset is not None:
        args.run_name = f"{args.dataset.split('/')[-1].split('.')[0]}" \
            f"_{args.model_type}-{args.preset}_dropout-{args.dropout_p}" \
            f"_rseed-{args.r_seed}_start-{time.strftime('%y%b%d-%H%M', time.localtime())}"
    else:
        args.run_name = f"{args.dataset.split('/')[-1].split('.')[0]}" \
            f"_{args.model_type}_n-{args.num_layers}_h-{args.num_heads}" \
            f"_d-{args.d_model}_dff-{args.d_ff}_dropout-{args.dropout_p}" \
            f"_rseed-{args.r_seed}_start-{time.strftime('%y%b%d-%H%M', time.localtime())}"

restore_args = " \\\n  ".join(sys.argv[1:])
if "--run-name" not in restore_args:
    restore_args += f" \\\n  --run-name {args.run_name}"

sbatch_executable = f"""#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 2-11:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu                             # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                          # Memory total in MB (for all cores)
#SBATCH -o slurm_files/slurm-%j.out        # File to which STDOUT + STDERR will be written, including job ID in filename
hostname
pwd
module load gcc/6.2.0 cuda/9.0
srun stdbuf -oL -eL {sys.executable} \\
  {sys.argv[0]} \\
  {restore_args} \\
  --restore {{restore}}
"""


####################
# SET RANDOM SEEDS #
####################

if args.restore is not None:
    # prevent from repeating batches/seed when restoring at intermediate point
    # script is repeatable as long as restored at same step
    # assumes restore arg of *[_/][step].pth
    args.r_seed += int(args.restore.split('_')[-1].split('/')[-1].split('.')[0])
    args.r_seed = args.r_seed % (2 ** 32 - 1)  # limit of np.random.seed

np.random.seed(args.r_seed)
torch.manual_seed(args.r_seed)
torch.cuda.manual_seed_all(args.r_seed)


def _init_fn(worker_id):
    np.random.seed(args.r_seed + worker_id)


#####################
# PRINT SYSTEM INFO #
#####################

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)

USE_CUDA = not args.no_cuda
device = torch.device("cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu")
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
    print(get_cuda_version())
    print("CuDNN Version ", get_cudnn_version())
print()

print("Run:", args.run_name)


#####################
# MAKE MODEL PARAMS #
#####################

fr = False
bert = False
bert_mask_freq = 0.15
bert_mask_proportion = (0.8, 0.1, 0.1)  # mask, random, keep
bert_params = {
    'attn_mask_type': 'seq',            # seq, bert, diag
    'attn_mask_layer': 'first',           # all, first, first_half
    'residual_layer': 'all',            # all, not_first, none
}
if args.model_type == 'transformer-fr':
    model_type = models.TransformerDecoderFR
    fr = True
elif args.model_type == 'BERT':
    model_type = models.UnconditionedBERT
    bert = True
    if 'masked' in args.preset:
        bert_params['attn_mask_type'] = 'seq,bert'
        if 'allmasked' in args.preset:
            bert_params['attn_mask_layer'] = 'all'
        elif 'halfmasked' in args.preset:
            bert_params['attn_mask_layer'] = 'first_half'
    if 'nokeep' in args.preset:
        bert_mask_freq = 0.15
        bert_mask_proportion = (0.8, 0.2, 0.0)
    elif 'allrandom' in args.preset:
        bert_mask_freq = 0.1
        bert_mask_proportion = (0.0, 1.0, 0.0)
        if 'masked' in args.preset:
            bert_mask_freq = 0.15
    elif 'allkeep' in args.preset:
        bert_mask_freq = 0.15
        bert_mask_proportion = (0.0, 0.0, 1.0)
        if 'masked' in args.preset:
            bert_params['attn_mask_type'] = 'seq,bert,diag'

            # block residual in first layer to prevent info leakage
            # TODO: other option: 'mask_first': masked residual connections on first layer
            bert_params['residual_layer'] = 'not_first'
else:
    model_type = models.TransformerDecoder


#############
# LOAD DATA #
#############

dataset = data_loaders.DoubleWeightedIndexedAntibodyDataset(
    batch_size=args.batch_size,
    working_dir=data_dir,
    dataset=args.dataset,
    matching=fr,
    unlimited_epoch=True,
    output_shape='NLC',
    output_types='encoder' if bert else 'decoder',
)
if bert:
    dataset = data_loaders.BERTPreprocessorDataset(
        dataset,
        mask_freq=bert_mask_freq,
        mask_proportion=bert_mask_proportion,
    )
loader = data_loaders.GeneratorDataLoader(
    dataset,
    num_workers=args.num_data_workers,
    pin_memory=True,
    worker_init_fn=_init_fn
)

if args.num_epochs is not None:
    args.num_iterations = math.ceil(args.num_epochs * dataset.n_eff / dataset.batch_size)


##############
# LOAD MODEL #
##############

if args.restore is not None:
    print("Restoring model from:", args.restore)
    checkpoint = torch.load(args.restore, map_location='cpu' if device.type == 'cpu' else None)
    dims = checkpoint['model_dims']
    hyperparams = checkpoint['model_hyperparams']
    if bert:
        hyperparams['bert']['attn_mask_type'] = hyperparams['bert'].pop('mask_type')
        hyperparams['bert']['attn_mask_layer'] = hyperparams['bert'].pop('mask_layer')
    trainer_params = checkpoint['train_params']
    model = model_type(dims=dims, hyperparams=hyperparams)
else:
    checkpoint = args.restore
    trainer_params = None
    dims = {
        'input': len(dataset.alphabet),
        'alphabet': len(dataset.output_alphabet),
    }
    hyperparams = {
        'transformer': {
            'd_model': args.d_model,
            'd_ff': args.d_ff,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'dropout_p': args.dropout_p,
            'pe_random_start': 'randomstart' in args.preset.split('-'),
        }
    }
    if bert:
        hyperparams['bert'] = bert_params
        # hyperparams['optimization'] = {'lr_factor': 2. / bert_mask_freq}
    model = model_type(dims=dims, hyperparams=hyperparams)
model.to(device)


################
# RUN TRAINING #
################

trainer = trainers.TransformerTrainer(
    model=model,
    data_loader=loader,
    params=trainer_params,
    snapshot_path=working_dir + '/snapshots',
    snapshot_name=args.run_name,
    snapshot_interval=args.num_iterations // 10,
    snapshot_exec_template=sbatch_executable,
    device=device,
    # logger=model_logging.Logger(log_interval=100, validation_interval=None),
    logger=model_logging.TensorboardLogger(
        log_interval=500,
        validation_interval=1000,
        generate_interval=5000,
        info_interval=10000,
        log_dir=working_dir + '/logs/' + args.run_name,
        print_output=True,
    ),
)
if args.restore is not None:
    trainer.load_state(checkpoint)

print()
print("Model:", model_type.__name__)
print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
print("Trainer:", trainer.__class__.__name__)
print("Training parameters:", json.dumps(
    {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
print("Dataset:", dataset.__class__.__name__)
print("Dataset parameters:", json.dumps(dataset.params, indent=4))
print("Num trainable parameters:", model.parameter_count())
print(f"Training for {args.num_iterations - model.step} iterations")

trainer.train(steps=args.num_iterations)

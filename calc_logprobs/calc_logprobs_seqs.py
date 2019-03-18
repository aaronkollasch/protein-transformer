#!/usr/bin/env python
import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import torch

sys.path.append("..")
import models
import trainers
import data_loaders

parser = argparse.ArgumentParser(description="Calculate the log probability of mutated sequences.")
parser.add_argument("--restore", type=str, default='', required=True,
                    help="Snapshot name for restoring the model")
parser.add_argument("--input", type=str, default='', required=True,
                    help="Directory and filename of the input data.")
parser.add_argument("--output", type=str, default='output', required=True,
                    help="Directory and filename of the output data.")
parser.add_argument("--num-samples", type=int, default=1,
                    help="Number of iterations to run the model.")
parser.add_argument("--batch-size", metavar='N', type=int, default=100,
                    help="Batch size.")
parser.add_argument("--dropout-p", type=float, default=0.,
                    help="Dropout p while sampling log p(x) (drop rate, not keep rate)")
parser.add_argument("--num-data-workers", type=int, default=0,
                    help="Number of workers to load data")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU evaluation")
args = parser.parse_args()

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
print()

print("Loading checkpoint")
checkpoint = torch.load(os.path.join('../snapshots', args.restore), map_location='cpu')

fr = False
bert = False
if checkpoint['model_type'] == 'transformer_decoder_fr':
    model_type = models.TransformerDecoderFR
    fr = True
elif checkpoint['model_type'] == 'bert_transformer_decoder':
    model_type = models.UnconditionedBERT
    bert = True
else:
    model_type = models.TransformerDecoder

print("Reading test data")
dataset = data_loaders.FastaDataset(
    batch_size=args.batch_size,
    working_dir='.',
    dataset=args.input,
    matching=fr,
    unlimited_epoch=False,
    output_shape='NLC',
    output_types='encoder' if bert else 'decoder',
)
if bert:
    dataset_params = checkpoint['dataset_params']
    dataset = data_loaders.BERTPreprocessorDataset(
        dataset,
        mask_freq=dataset_params['mask_freq'],
        mask_proportion=[0., 0., 1.],
    )
loader = data_loaders.GeneratorDataLoader(
    dataset,
    num_workers=args.num_data_workers,
    pin_memory=True,
)

print("Initializing model")
dims = checkpoint['model_dims']
hyperparams = checkpoint['model_hyperparams']
hyperparams['transformer']['dropout_p'] = args.dropout_p
# if bert:
#     hyperparams['bert']['attn_mask_type'] = 'seq,diag'
model = model_type(dims=dims, hyperparams=hyperparams)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print("Num parameters:", model.parameter_count())

trainer = trainers.TransformerTrainer(
    model=model,
    data_loader=loader,
    device=device,
)

print()
print("Model:", model_type.__name__)
print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
print("Trainer:", trainer.__class__.__name__)
print("Training parameters:", json.dumps(
    {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
print("Dataset:", dataset.__class__.__name__)
print("Dataset parameters:", json.dumps(dataset.params, indent=4))
print("Num trainable parameters:", model.parameter_count())
print()
output = trainer.test(loader, model_eval=False, dataset_eval=False, num_samples=args.num_samples)
output = pd.DataFrame(output, columns=output.keys())
output.to_csv(args.output, index=False)
print("Done!")

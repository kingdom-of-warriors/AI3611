# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx

import data

# My module code
from models import RNNModel, SpeechAwareTransformer, TransformerModel
from utils import batchify, export_onnx, generate_model_save_path
from eval import evaluate
from train import train

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/gigaspeech',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='ckpt/',
                    help='directory to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda: print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
elif args.model == 'SpeechAwareTransformer':
    model = SpeechAwareTransformer(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()

print ("Vocabulary Size: ", ntokens)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print ("Total number of model parameters: {:.2f}M".format(num_params*1.0/1e6))

###############################################################################
# Training code
##############################################################################

best_val_loss = None
lr = args.lr
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(model, args.model, corpus, train_data, criterion, lr, args.clip, args.bptt, args.batch_size, epoch, args.log_interval, args.dry_run)
        val_loss = evaluate(model, args.model, corpus, val_data, args.batch_size, args.bptt, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        
        # generate save path of present model
        save_path = generate_model_save_path(args)
        with open(save_path, 'wb') as f:
            torch.save(model, f)
            print(f"| Model saved to {save_path}")
        
        # save the best model
        if not best_val_loss or val_loss < best_val_loss:
            best_save_path = generate_model_save_path(args, suffix="best")
            with open(best_save_path, 'wb') as f:
                torch.save(model, f)
                print(f"| Best model saved to {best_save_path}")
            
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# fetch and load the best model
best_model_path = generate_model_save_path(args)
print(f"Loading best model from {best_model_path}")
with open(best_model_path, 'rb') as f:
    model = torch.load(f, weights_only=False)   # to support pytorch 2.6.0
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(model, args.model, corpus, test_data, args.batch_size, args.bptt, criterion)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(model, device, args.onnx_export, args.batch_size, args.bptt)
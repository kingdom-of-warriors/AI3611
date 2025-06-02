# Word-level Language Modeling using RNN and Transformer

This example trains a multi-layer RNN (Elman, GRU, or LSTM) or Transformer on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA.
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA.
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs.
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA.

python generate.py                         # Generate samples from the trained LSTM model.
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```
## Best Model
To get the best `ppl` in test dataset, you should run
```bash
python main.py --cuda --model Transformer --emsize 512 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32 # PPL 133.05
```
Model size is $38.93M$.

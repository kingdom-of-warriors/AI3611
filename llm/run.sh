# influence of bptt
# RNN LSTM GRU
python main.py --cuda --model RNNModel --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 32 --epochs 40 --lr 1.0 --batch_size 32
python main.py --cuda --model RNNModel --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 64 --epochs 40 --lr 1.0 --batch_size 32
python main.py --cuda --model RNNModel --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32
python main.py --cuda --model RNNModel --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 256 --epochs 40 --lr 1.0 --batch_size 32    
# Transformer
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 32 --epochs 40 --lr 1.0 --batch_size 32 # 146.88
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 64 --epochs 40 --lr 1.0 --batch_size 32 # 137.95
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32 # 135.74
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 256 --epochs 40 --lr 1.0 --batch_size 32 # 160.98
# SpeechAwareTransformer
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 32 --epochs 10 --lr 1.0 --batch_size 32 # 1.21
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 64 --epochs 10 --lr 1.0 --batch_size 32 # 1.10
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 10 --lr 1.0 --batch_size 32 # 1.05
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 256 --epochs 10 --lr 1.0 --batch_size 32 # 1.06
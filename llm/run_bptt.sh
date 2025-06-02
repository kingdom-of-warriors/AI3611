# influence of bptt
# RNN LSTM GRU
python main.py --cuda --model GRU --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 32 --epochs 40 --lr 10.0 # 138.30
python main.py --cuda --model GRU --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 64 --epochs 40 --lr 10.0 # 137.00
python main.py --cuda --model GRU --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 128 --epochs 40 --lr 10.0 # 135.33
python main.py --cuda --model GRU --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 256 --epochs 40 --lr 10.0 # 135.26  

python main.py --cuda --model LSTM --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 32 --epochs 40 --lr 10.0 # 135.97
python main.py --cuda --model LSTM --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 64 --epochs 40 --lr 10.0 # 133.11
python main.py --cuda --model LSTM --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 128 --epochs 40 --lr 10.0 # 133.58
python main.py --cuda --model LSTM --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 256 --epochs 40 --lr 10.0 # 133.79   
# Transformer
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 32 --epochs 40 --lr 1.0 # 146.88
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 64 --epochs 40 --lr 1.0 # 137.95
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 # 135.74
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 256 --epochs 40 --lr 1.0 # 160.98
# SpeechAwareTransformer
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 32 --epochs 10 --lr 1.0 --batch_size 32 # 1.21
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 64 --epochs 10 --lr 1.0 --batch_size 32 # 1.10
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 10 --lr 1.0 --batch_size 32 # 1.05
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 256 --epochs 10 --lr 1.0 --batch_size 32 # 1.06
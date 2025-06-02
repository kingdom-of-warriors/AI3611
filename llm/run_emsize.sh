# influence of bptt
# RNN LSTM GRU
python main.py --cuda --model LSTM --emsize 128 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 128 --epochs 40 --lr 10.0  # 134.75
python main.py --cuda --model LSTM --emsize 256 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 128 --epochs 40 --lr 10.0  # 133.53
python main.py --cuda --model LSTM --emsize 384 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 128 --epochs 40 --lr 10.0  # 133.58
python main.py --cuda --model LSTM --emsize 512 --nhid 256 --dropout 0.2 --nlayers 2 --bptt 128 --epochs 40 --lr 10.0  # 133.12
# Transformer
python main.py --cuda --model Transformer --emsize 128 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32 # 149.36
python main.py --cuda --model Transformer --emsize 256 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32 # 139.67
python main.py --cuda --model Transformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32 # 133.43
python main.py --cuda --model Transformer --emsize 512 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 40 --lr 1.0 --batch_size 32 # 133.05
# SpeechAwareTransformer
python main.py --cuda --model SpeechAwareTransformer --emsize 128 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 10 --lr 1.0 --batch_size 32  # 1.05
python main.py --cuda --model SpeechAwareTransformer --emsize 256 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 10 --lr 1.0 --batch_size 32  # 1.05
python main.py --cuda --model SpeechAwareTransformer --emsize 384 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 10 --lr 1.0 --batch_size 32  # 1.05
python main.py --cuda --model SpeechAwareTransformer --emsize 512 --nhid 256 --dropout 0.2 --nlayers 8 --bptt 128 --epochs 10 --lr 1.0 --batch_size 32 # 1.05
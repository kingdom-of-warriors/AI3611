import torch
import os

def batchify(data, bsz, device="cuda"):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def export_onnx(model, device, path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(path)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def get_batch(bptt, source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

def generate_model_save_path(args, suffix=None):
    """
    根据模型类型和超参数生成保存路径
    格式: "模型类型/各种超参数的集合.pt"
    
    参数:
        args: 命令行参数对象，包含各种模型超参数
        suffix: 可选，一些有必要的后缀
        
    返回:
        保存模型的完整路径
    """
    # 确定模型类型文件夹
    if args.model == 'Transformer':
        model_type = 'transformer'
    elif args.model == 'SpeechAwareTransformer':
        model_type = 'speech_transformer'
    else:
        model_type = args.model.lower()  # RNN_TANH, RNN_RELU, LSTM, GRU
    
    # 构建超参数字符串
    params = []
    
    # 添加共同参数
    params.append(f"emb{args.emsize}")
    params.append(f"hid{args.nhid}")
    params.append(f"lay{args.nlayers}")
    params.append(f"dp{args.dropout}")
    params.append(f"bptt{args.bptt}")
    
    # 添加特定模型参数
    if args.model in ['Transformer', 'SpeechAwareTransformer']:
        params.append(f"head{args.nhead}")
    
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        params.append("tied" if args.tied else "untied")
    
    # 添加其他可能有用的参数
    params.append(f"lr{args.lr}")
    params.append(f"bs{args.batch_size}")
    
    # 如果指定了后缀，添加到文件名中
    if suffix is not None: params.append(f"{suffix}")
    
    # 构建文件名
    filename = "_".join(params) + ".pt"
    
    # 构建完整路径
    save_dir = os.path.join(args.save, model_type)
    os.makedirs(save_dir, exist_ok=True)
    
    return os.path.join(save_dir, filename)
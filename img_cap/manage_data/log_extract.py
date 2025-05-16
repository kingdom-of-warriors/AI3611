import re

def extract_train_log_data(log_file_path):
    """
    从训练日志中提取 train_loss 和 val_bleu1。
    Args:
        log_file_path (str): 日志文件的路径。
    Returns:
        tuple: 包含两个列表的元组 (train_losses, val_bleu1s)
    """
    train_losses = []
    val_bleu1s = []

    # 例如: [2025-05-08 11:48:50] Epoch 1/30, train_loss: 4.135, val_bleu1: 0.583, val_bleu4: 0.114
    regex = r"Epoch \d+/\d+, train_loss: ([\d.]+), val_bleu1: ([\d.]+),"

    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(regex, line)
            if match:
                train_loss = float(match.group(1))
                val_bleu1 = float(match.group(2))
                train_losses.append(train_loss)
                val_bleu1s.append(val_bleu1)

    return train_losses, val_bleu1s

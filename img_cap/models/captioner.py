import torch
import torch.nn as nn
from .decoder import DecoderWithAttention
from .resnet_encoder import Encoder
from .vit_encoder import VitEncoder

class Res_Captioner(nn.Module):
    def __init__(self, encoded_image_size, attention_dim, embed_dim,
        decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, **kwargs):
        super().__init__()
        self.encoder = Encoder(encoded_image_size=encoded_image_size)
        self.decoder = DecoderWithAttention(attention_dim, embed_dim,
            decoder_dim, vocab_size, encoder_dim, dropout)

    def forward(self, images, encoded_captions, caption_lengths, teacher_forcing_ratio=1.0):
        """
        :param images: [b, 3, h, w]
        :param encoded_captions: [b, max_len]
        :param caption_lengths: [b,]
        :param teacher_forcing_ratio: probability for teacher forcing
        :return:
        """
        encoder_out = self.encoder(images)
        decoder_out = self.decoder(encoder_out, encoded_captions,
                                   caption_lengths.unsqueeze(1),
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_out

    def sample(self, images, startseq_idx, endseq_idx=-1, max_len=40,
               method='beam', return_alpha=False):
        encoder_out = self.encoder(images)
        return self.decoder.sample(encoder_out=encoder_out,
            startseq_idx=startseq_idx, endseq_idx=endseq_idx, max_len=max_len,
            method=method, return_alpha=return_alpha)


class Vit_Captioner(nn.Module):
    def __init__(self, encoded_image_size, attention_dim, embed_dim,
        decoder_dim, vocab_size, encoder_dim=768, dropout=0.5, **kwargs):
        super().__init__()
        self.encoder = VitEncoder(encoded_image_size=encoded_image_size)
        self.decoder = DecoderWithAttention(attention_dim, embed_dim,
            decoder_dim, vocab_size, encoder_dim, dropout)

    def forward(self, images, encoded_captions, caption_lengths, teacher_forcing_ratio=1.0):
        """
        :param images: [b, 3, h, w]
        :param encoded_captions: [b, max_len]
        :param caption_lengths: [b,]
        :param teacher_forcing_ratio: probability for teacher forcing
        :return:
        """
        encoder_out = self.encoder(images)
        decoder_out = self.decoder(encoder_out, encoded_captions,
                                   caption_lengths.unsqueeze(1),
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_out

    def sample(self, images, startseq_idx, endseq_idx=-1, max_len=40,
               method='beam', return_alpha=False):
        encoder_out = self.encoder(images)
        return self.decoder.sample(encoder_out=encoder_out,
            startseq_idx=startseq_idx, endseq_idx=endseq_idx, max_len=max_len,
            method=method, return_alpha=return_alpha)
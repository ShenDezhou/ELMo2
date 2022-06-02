# -*- coding: UTF-8 -*-

import torch

from config import Config
from model.ELMo import ConvTokenEmbedder
from utils import Vocab

class ELMo_Util:
    def __init__(self, vocab_file="trained_model/elmo_bilm/char.dic", token_model="trained_model/elmo_bilm/token_embedder.pth"):
        BOS_TOKEN = '<BOS>'
        EOS_TOKEN = '<EOS>'
        PAD_TOKEN = '<PAD>'
        BOW_TOKEN = '<BOW>'
        EOW_TOKEN = '<EOW>'

        configs = Config()
        charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN}
        with open(vocab_file,"r", encoding="utf-8") as f:
            for line in f:
                charset.add(line.strip('\n'))
        self.vocab_c = Vocab(list(charset), min_freq=1)

        self.token_embedder = ConvTokenEmbedder(
                    self.vocab_c,
                    configs.char_embedding_dim,
                    configs.char_conv_filters,
                    configs.num_highways,
                    configs.projection_dim
                )

        self.token_embedder.load_state_dict(torch.load(token_model))
        self.char_embeddings = self.token_embedder.char_embeddings


    def encode(self, sentence):
        char_ind = self.vocab_c.convert_tokens_to_ids(list(sentence))
        encoded = self.char_embeddings(torch.tensor(char_ind))
        return encoded

if __name__ == "__main__":
    elmo_util = ELMo_Util()
    res = elmo_util.encode("还好")
    print(res)
import aoareader as reader
import torch

data_file = "data/preprocessed.pt"

data = torch.load(data_file)

vocab_dict = data['dict']
train_data = data['train']
valid_data = data['valid']

valid_dataset = reader.Dataset(valid_data, 32, False)

model = reader.AoAReader(vocab_dict, 0.2, 384, 384)


(docs, docs_len), (querys, querys_len), _ = valid_dataset[0]

beta = model(docs, docs_len, querys, querys_len)

import numpy as np

beta = beta.data.numpy()
print(np.sum(beta, axis=2))


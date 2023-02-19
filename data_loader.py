import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
import pandas as pd

from hparams import hparams 
from embed.ft_embed import FT
from embed.vader_embed import Vader

#embedder = FT()
embedder = Vader()

class Dataset(Dataset):
    def __init__(self, lyrics, labels):
        self.lyrics = lyrics
        self.labels = labels

    def __getitem__(self, index):
        return self.lyrics[index], self.labels[index]

    def __len__(self):
        return len(self.lyrics)
    
def load_dataset(set_name):
    lyrics_list = []
    label_list = []
    df = pd.read_csv(hparams.dataset_path + set_name + '.csv').loc[:, ['lyrics', hparams.label_name]]

    for index in range(len(df)):
        lyrics = embedder.lyrics_to_vec(df.iloc[index]['lyrics'])
        label = df.iloc[index][hparams.label_name]
        lyrics_list.append(lyrics)
        label_list.append(label)

    label_list = torch.LongTensor(label_list)
    return lyrics_list, label_list

def get_dataloader(hparams):
    if hparams.status == 'train':
        lyrics_train, label_train = load_dataset('train')
        lyrics_valid, label_valid = load_dataset('valid')

        lyrics_train = np.stack(lyrics_train)
        lyrics_valid = np.stack(lyrics_valid)

        label_train = np.stack(label_train)
        label_valid = np.stack(label_valid)

        train_set = Dataset(lyrics_train, label_train)
        valid_set = Dataset(lyrics_valid, label_valid)

        train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=hparams.batch_size, shuffle=False, drop_last=True)

    lyrics_test, label_test = load_dataset('test')
    lyrics_test = np.stack(lyrics_test)
    label_test = np.stack(label_test)
    test_set = Dataset(lyrics_test, label_test)
    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, drop_last=True)

    if hparams.status == 'train':
        return train_loader, valid_loader, test_loader
    else:
        return test_loader

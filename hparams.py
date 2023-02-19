'''
hparams.py

A file that sets hyper parameters for feature extraction and training.
You can change parameters using argument.
'''
import argparse

class HParams(object):
    def __init__(self):
        self.dataset_path = 'dataset/'
        self.label_name = 'valence_bin' # 'arousal_bin','valence_bin'
        self.status = 'eval' # 'train', 'eval'
        self.arousal_model_path = 'models/arousal_vader.pt'
        self.valence_model_path = 'models/valence_vader.pt'
        self.model_save_path = 'modes/latest.pt'

        # Training Parameters
        self.device = 1  # 0: CPU, 1: GPU0, 2: GPU1, ...
        self.batch_size = 1
        self.num_epochs = 100
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 3
        self.max_len = 50
        self.emb_dim = 100
        self.lstm_hid_dim = 40

	# Function for parsing argument and set hyper parameters
    def parse_argument(self, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(hparams, var)
            argument = '--' + var
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args()
        for var in vars(self):
            setattr(hparams, var, getattr(args, var))

        if print_argument:
            print('-------------------------')
            print('Hyper Parameter Settings')
            print('-------------------------')
            for var in vars(self):
                value = getattr(hparams, var)
                print(var + ': ' + str(value))
            print('-------------------------')

hparams = HParams()
hparams.parse_argument()

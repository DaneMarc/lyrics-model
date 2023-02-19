'''
train_test.py

A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
# torch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
# private
import data_loader
from hparams import hparams
import utils

from models.ft_model import MultiNet
from models.vader_model import DeathStar


class Runner(object):
    def __init__(self, hparams):
        self.device = hparams.device
        #self.model = MultiNet(hparams)
        self.model = DeathStar(hparams)

        if hparams.status == 'eval':
            if hparams.arousal_model_path and hparams.label_name == 'arousal_bin':
                self.model.load_state_dict(torch.load(hparams.arousal_model_path))
            elif hparams.valence_model_path and hparams.label_name == 'valence_bin':
                self.model.load_state_dict(torch.load(hparams.valence_model_path))

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def accuracy(self, source, target):
        source = source.max(1)[1].long().cpu()
        target = target.long().cpu()
        correct = (source == target).sum().item()
        return correct/float(source.size(0))

    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode == 'train' else self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        
        model, floatTensor, longTensor = utils.set_device(self.model, self.device)

        for lyrics, labels in dataloader:
            lyrics = lyrics.type(floatTensor)
            labels = labels.type(longTensor)
                        
            prediction = self.model(lyrics)
            loss = self.criterion(prediction, labels)
            acc = self.accuracy(prediction, labels)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += prediction.size(0)*loss.item()
            epoch_acc += prediction.size(0)*acc

        epoch_loss = epoch_loss/len(dataloader.dataset)
        epoch_acc = epoch_acc/len(dataloader.dataset)

        return epoch_loss, epoch_acc

    # Early stopping function for given validation loss
    def early_stop(self, loss):
        self.scheduler.step(loss)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop


def main():
    runner = Runner(hparams)

    if hparams.status == 'train':
        train_loader, valid_loader, test_loader = data_loader.get_dataloader(hparams)

        for epoch in range(hparams.num_epochs):
            train_loss, train_acc = runner.run(train_loader, 'train')
            valid_loss, valid_acc = runner.run(valid_loader, 'eval')

            print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
                (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))

            if runner.early_stop(valid_loss):
                break

        print("Training Finished")
    else:
        test_loader = data_loader.get_dataloader(hparams)

    test_loss, test_acc = runner.run(test_loader, 'eval')
    print("Test Loss: %.4f" % test_loss)
    print("Test Accuracy: %.2f%%" % (100*test_acc))

    if hparams.status == 'train':
        torch.save(runner.model.state_dict(), hparams.model_save_path)

if __name__ == '__main__':
    main()

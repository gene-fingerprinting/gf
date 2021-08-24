import const
import numpy as np
import pandas as pd

from os.path import join, exists
from utility import load_obj, save_obj
import torch
import torch.nn as nn
import torch.nn.functional as F

saved_model = join(const.MODEL_DIR, 'saved_model_DF.pkl')

# Model Hyperparameters
filter_size = 8             # [2 ... 16]
pool_size = 8               # [2 ... 16]
stride_size = 4             # [2 ... 16]
dropout = 0.1               # [0.1 .. 0.8]
filter_num_b1 = [32, 32]    # [8 ... 64]
filter_num_b2 = [64, 64]    # [32 ... 128]
filter_num_b3 = [128, 128]  # [64 ... 256]
filter_num_b4 = [256, 256]  # [128 ... 512]
hidden_size_fc1 = 512       # [256 ... 2048]
hidden_size_fc2 = 512       # [256 ... 2048]
dropout_fc1 = 0.7           # [0.1 .. 0.8]
dropout_fc2 = 0.5           # [0.1 .. 0.8]

# Training Hyperparameters
optim = torch.optim.Adamax  # [Adam, Adamax, RMSProp, SGD]
learning_rate = 0.002       # [0.001 ... 0.01]
training_epochs = 30        # [10 ... 50]
batch_size = 128            # [16 ... 256]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DF:
    def __init__(self, input, classes, mode, quantile=None):
        self.model = DFModel(input, classes).to(device)
        self.input = input
        self.mode = mode
        self.quantile = quantile

    def load(self, X, y=None):
        # Pad to input size
        for i in range(len(X)):
            X_i = X[i]
            if len(X_i) < self.input:
                # Shorter than input size
                X_i = np.append(X_i, np.zeros(self.input - len(X_i)))
            else:
                # Longer than input size
                X_i = X_i[:self.input]
            X[i] = X_i

        X = np.array(X)
        X = X.astype('float32')
        X = X[:, :, np.newaxis]
        X = torch.from_numpy(X).permute(0, 2, 1)
        if y is None:
            return X
        else:
            y = np.array(y)
            y = y.astype('int64')
            y = torch.from_numpy(y)
            return X, y
    
    def train(self, X_train, y_train, verbose=False):
        X_train, y_train = self.load(X_train, y_train)
        if not exists(saved_model):
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim(self.model.parameters(), learning_rate)
            data_loader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True)

            for epoch in range(training_epochs):
                for _, (batch_x, batch_y) in enumerate(data_loader):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if verbose:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, training_epochs, loss.item()))

            save_obj(self.model, saved_model)
        else:
            self.model = load_obj(saved_model)

        if self.mode == const.MODE_OW:
            self.threshold(X_train, y_train)
    
    def threshold(self, X_train, y_train):
        scores = self.predict_score(X_train)
        max_score = [score.max() for score in scores]
        max_removed = []
        y = np.array(y_train)
        ys = np.unique(y_train)
        for y_i in ys:
            max_removed += [max([score.max() for score in y_scores[ys != y_i]]) for y_scores in scores[y == y_i]]
        
        fitter = pd.DataFrame({'y': y_train, 'max_score': max_score, 'max_removed': max_removed})
        thresholds = {}
        for y_i in ys:
            y_scores = fitter[fitter['y'] == y_i]
            upper_score = y_scores['max_score'].quantile(q=self.quantile)
            lower_score = y_scores['max_removed'].quantile(q=1-self.quantile)
            thresholds[y_i] = (upper_score + lower_score) / 2
        self.thresholds = thresholds
    
    def test(self, X_test, y_test):
        X_test, y_test = self.load(X_test, y_test)
        if self.mode == const.MODE_CW:
            self.accuracy(X_test, y_test)
        else:
            self.metrics(X_test, y_test)
    
    def accuracy(self, X_test, y_test):
        y_predict = self.predict(X_test, y_test)
        correct = sum(y_predict == np.array(y_test))
        accuracy = correct / len(y_test)
        print('Accuracy: {:.2f} %'.format(100 * accuracy))
    
    def metrics(self, X_test, y_test):
        scores = self.predict_score(X_test)
        y_predict = [score.argmax() if score.max() > self.thresholds[score.argmax()] else const.MON_NUM for score in scores]

        result = pd.DataFrame({'truth': y_test, 'predict': y_predict})
        TP = sum((result['truth'] < const.MON_NUM) & (result['predict'] < const.MON_NUM))
        FN = sum((result['truth'] < const.MON_NUM) & (result['predict'] == const.MON_NUM))
        FP = sum((result['truth'] >= const.MON_NUM) & (result['predict'] < const.MON_NUM))
        TN = sum((result['truth'] >= const.MON_NUM) & (result['predict'] == const.MON_NUM))

        TPR = TP / (TP + FN)
        FNR = FN / (TP + FN)
        FPR = FP / (FP + TN)
        Precision = TP / (TP + FP)
        print('TPR: {:.2f} %'.format(100 * TPR), end='\t')
        print('FNR: {:.2f} %'.format(100 * FNR), end='\t')
        print('FPR: {:.2f} %'.format(100 * FPR), end='\t')
        print('Precision: {:.2f} %'.format(100 * Precision))
    
    def predict(self, X_test, y_test):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=batch_size)
        y_predict = None
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(device)
                outputs = self.model(batch_x)
                _, batch_y = torch.max(outputs.data, 1)
                if y_predict is None:
                    y_predict = batch_y
                else:
                    y_predict = torch.cat([y_predict, batch_y])

        return y_predict.cpu().numpy()

    def predict_score(self, X_test, y_test):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=batch_size)
        scores = None
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(device)
                score = self.model(batch_x)
                if scores is None:
                    scores = score
                else:
                    scores = torch.cat([scores, score])

        return scores.cpu().numpy()


class Conv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride)

    def forward(self, x):
        net = x

        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MaxPool1dPadSame(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class DFModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DFModel, self).__init__()

        res_channels = input_size
        for _ in range(4):
            res_channels = int(np.ceil(res_channels / pool_size))

        self.block1 = nn.Sequential(
            Conv1dPadSame(1, filter_num_b1[0], filter_size, 1),
            nn.BatchNorm1d(filter_num_b1[0]),
            nn.ELU(),
            Conv1dPadSame(filter_num_b1[0], filter_num_b1[1], filter_size, 1),
            nn.BatchNorm1d(filter_num_b1[1]),
            nn.ELU(),
            MaxPool1dPadSame(pool_size, stride_size),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            Conv1dPadSame(filter_num_b1[1], filter_num_b2[0], filter_size, 1),
            nn.BatchNorm1d(filter_num_b2[0]),
            nn.ReLU(),
            Conv1dPadSame(filter_num_b2[0], filter_num_b2[1], filter_size, 1),
            nn.BatchNorm1d(filter_num_b2[1]),
            nn.ReLU(),
            MaxPool1dPadSame(pool_size, stride_size),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            Conv1dPadSame(filter_num_b2[1], filter_num_b3[0], filter_size, 1),
            nn.BatchNorm1d(filter_num_b3[0]),
            nn.ReLU(),
            Conv1dPadSame(filter_num_b3[0], filter_num_b3[1], filter_size, 1),
            nn.BatchNorm1d(filter_num_b3[1]),
            nn.ReLU(),
            MaxPool1dPadSame(pool_size, stride_size),
            nn.Dropout(dropout)
        )
        self.block4 = nn.Sequential(
            Conv1dPadSame(filter_num_b3[1], filter_num_b4[0], filter_size, 1),
            nn.BatchNorm1d(filter_num_b4[0]),
            nn.ReLU(),
            Conv1dPadSame(filter_num_b4[0], filter_num_b4[1], filter_size, 1),
            nn.BatchNorm1d(filter_num_b4[1]),
            nn.ReLU(),
            MaxPool1dPadSame(pool_size, stride_size),
            nn.Dropout(dropout)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(filter_num_b4[1] * res_channels, hidden_size_fc1),
            nn.BatchNorm1d(hidden_size_fc1),
            nn.ReLU(),
            nn.Dropout(dropout_fc1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size_fc1, hidden_size_fc2),
            nn.BatchNorm1d(hidden_size_fc2),
            nn.ReLU(),
            nn.Dropout(dropout_fc2)
        )
        self.predict = nn.Linear(hidden_size_fc2, num_classes)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        out = self.predict(out)
        return out

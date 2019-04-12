import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import random
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
    """
    c1_out = convolutional layer 1 filter count
    c2_out = convolutional layer 2 filter count
    out_size = number of labels
    """

    def __init__(self, c1_out, c2_out, l1_out, l2_out, out_size, kernel_size):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, c1_out, kernel_size)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(c2_out * kernel_size * kernel_size, l1_out)
        self.fc2 = nn.Linear(l1_out, l2_out)
        self.fc3 = nn.Linear(l2_out, out_size)

    def forward(self, x, pool_size):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), pool_size)
        x = F.max_pool2d(F.relu(self.conv2(x)), pool_size)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save_model_weights(self, suffix):
        torch.save(self.state_dict(), ''.join([suffix, '.pt']))

    def load_model_weights(self, model_name):
        self.load_state_dict(torch.load(model_name))


def one_hot_y(y, size):
    vec_y = []
    for idx in y:
        yy = np.zeros(size)
        yy[int(idx)] = 1
        vec_y.append(yy)
    return np.array(vec_y)


def import_data(saved_arr):
    arr = np.load(saved_arr)
    x = arr[:, :-1]
    y = arr[:, -1]
    y = one_hot_y(y, 2)
    return x, y

def sample_batch(x, y, batch_size):

    inds = random.sample(range(len(x)),batch_size)
    batch_x = [x[i] for i in inds]
    batch_y = [y[i] for i in inds]

    return torch.FloatTensor(batch_x), torch.FloatTensor(batch_y)

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

if __name__ == "__main__":
    arr_pos = sys.argv[1]
    arr_neg = sys.argv[2]

    x_pos, y_pos = import_data(arr_pos)
    x_neg, y_neg = import_data(arr_neg)

    x = np.concatenate((x_pos, x_neg))
    y = np.concatenate((y_pos, y_neg))

    dataset = torch.utils.data.Dataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size)

    data_splits = [np.floor(0.7*len(x)), np.floor(0.22*len(x)), len(x) - np.floor(0.7*len(x)) - np.floor(0.2*len(x))]

    x_training, x_val, x_test = torch.utils.data.random_split(x, data_splits)
    y_training, y_val, y_test = torch.utils.data.random_split(y, data_splits)

# Network Params: c1_out, c2_out, l1_out, l2_out, out_size, kernel_size

    net = Net(6, 6, 200, 100, 2)

    batch_size = 32
    epochs = 100
    net.objective = nn.CrossEntropyLoss()
    optimizer = optim.adam(net.parameters(), lr = 0.0001)

    training_dataset = torch.utils.data.Dataset(x_training, y_training)
    loader = torch.utils.data.DataLoader(training_dataset, batch_size, collate_fn=collate_wrapper, pin_memory=True)

    for batch_ndx, sample in enumerate(loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())


    #
    # for x in range(epochs):
    #     print("Epoch: ", x)
    #     for y in range(np.floor(len(x_training)/batch_size)):
    #
    #         batch_x, batch_y = sample_batch(x_training, y_training, batch_size)
    #
    #         optimizer.zero_grad()
    #         output = net(batch_x, 2)
    #         loss = net.objective(output, batch_y)
    #         loss.backward()
    #         optimizer.step()
    #
    # for x in




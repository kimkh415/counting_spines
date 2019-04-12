"""
Authors: Kwanho Kim, Saideep Gona, Jinke Liu

Contains code for training convolutional neural networks for predicting 
dendritic spine presence from pre-constructed training patches.
"""
import sys
import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import random
import torchvision
import torchvision.transforms as transforms
import datetime
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

class Net(nn.Module):
    """
    c1_out = convolutional layer 1 filter count
    c2_out = convolutional layer 2 filter count
    out_size = number of labels
    """

    def __init__(self, c1_out, c2_out, l1_out, l2_out, out_size, kernel_size, patch_size, pool_size, pad):
        super(Net, self).__init__()
        # 1 input image channel

        self.conv1 = nn.Conv2d(1, c1_out, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size, padding=pad)

        self.pool_size = pool_size
        self.convout_size = int(c2_out * (patch_size/pool_size**2)**2)
        # self.convout_size = 3600
 
        print(self.convout_size, " size of convolution output")

        self.fc1 = nn.Linear(self.convout_size , l1_out)
        self.fc2 = nn.Linear(l1_out, l2_out)
        self.fc3 = nn.Linear(l2_out, out_size)

    def forward(self, x):
        # print(type(x))
        # Convolutions + Pooling
        c1 = F.relu(self.conv1(x))
        # print(c1.shape, " c1")
        p1 = F.max_pool2d(c1, self.pool_size)
        # print(p1.shape, " p1")
        c2 = F.relu(self.conv2(p1))
        # print(c2.shape, " c2")
        p2 = F.max_pool2d(c2, self.pool_size)
        # print(p2.shape, " p2")

        # Fully Connected
        flat = p2.view(-1, self.convout_size)
        # print(flat.shape, " flat")
        f1 = F.relu(self.fc1(flat))
        # print(f1.shape, " f1")l3
        f2 = F.relu(self.fc2(f1))
        f3 = self.fc3(f2)
        return f3

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save_model_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

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
    x = np.load(saved_arr + "_x.npy")
    y = np.load(saved_arr + "_y.npy")
    # x = arr[:,:, :-1]
    # y = arr[:,:, -1]
    # y = one_hot_y(y, 2)
    return x, y

def sample_batch(x, y, batch_size):

    all_inds = np.arange(len(x))

    sample_inds = np.random.choice(all_inds, batch_size)

    return x[sample_inds], y[sample_inds]


def pc(decimal):

    percent = decimal * 100
    str_pc = str(percent)

    return str_pc[0:5] + "%"

def epoch_loss_error(model, set_x, set_y):

    # print("epoch losses")
    forward_out = model.forward(set_x)
    # print(forward_out)
    
    _, preds = torch.max(forward_out,1)
    overlap = torch.eq(preds, set_y)
    # print(overlap)
    accuracy = float(torch.sum(overlap))/len(overlap)

    loss_out = model.objective(forward_out, set_y)
    # print(loss_out)
    return float(loss_out), (1-accuracy), ~overlap

def rand_split_data(x, y, p):
    """ 
    Randomly split data into train,val,test maintaining x-y mapping

    :param x: X dataset (examples)
    :param y: y dataset (labels)
    :param p: tuple of (train fraction, val fraction). Test fraction is remainder
    :return: all partitioned datasets (train, val, test) for (x,y)
    """
    shuffle_inds = np.random.shuffle(np.arange(len(x)))
    shuff_x = x[shuffle_inds]
    shuff_y = y[shuffle_inds]

    # Split shuffled data
    xl = len(x)
    data_splits = [int(p[0]*xl), int(p[1]*xl), xl - int(p[0]*xl) - int(p[1]*xl)]
    x_training, x_val, x_test = torch.split(x, data_splits)
    y_training, y_val, y_test = torch.split(y, data_splits)

    return x_training, x_val, x_test, y_training, y_val, y_test

def record_training(data, metadata, net, wrong_tests, correct_labels):
    """ 
    Save plots and associated metadata for a given training session

    :param data: Dictionary containing loss and error trajectories for train/val
    :param metadata: Dictionary containing various metadata values for the current run
    :param wrong_tests: Tensor containing incorrectly labeled patches 
    :param net: Trained network object 
    """
    timestamp = "".join(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(" "))
    timestamp = timestamp.replace(":", "_")
    cwd = os.getcwd()

    # Create timestamped directory
    if not os.path.exists(Path(cwd + "/training_sessions/")):
        os.makedirs(Path(cwd + "/training_sessions/"))
    
    save_dir = Path(cwd  + "/training_sessions/" + timestamp + "/")
    os.makedirs(save_dir)

    # Output metadata
    metadata_text = [
        "Patch Size: " + str(metadata["Patch Size"]),
        "Batch Size: " + str(metadata["Batch Size"]),
        "Learning Rate: " + str(metadata["Learning Rate"]),
        "Kernel Size: " + str(metadata["Kernel Size"]),
        "Epochs: " + str(metadata["Epochs"]),
        "Test Loss: " + str(metadata["Test Loss"]),
        "Test Error: " + str(metadata["Test Error"]),
        "\n"
    ]

    meta_file = Path(cwd  + "/training_sessions/" + timestamp + "/metadata.txt")
    with open(meta_file, "w") as m:
        m.write("\n".join(metadata_text))
        print(net, file=m)

    # Save model weights
    weights_file = Path(cwd  + "/training_sessions/" + timestamp + "/weights.pt")
    net.save_model_weights(weights_file)

    # Save images of incorrectly labeled test sets
    # print(wrong_tests.shape, " wrong tests")
    wrong_tests_np = wrong_tests.numpy()

    wrong_dir = Path(cwd  + "/training_sessions/" + timestamp + "/incorrect_labelings/")
    os.makedirs(wrong_dir)

    for x in range(len(wrong_tests)):
        sub_image = wrong_tests_np[x]. reshape((metadata["Patch Size"], metadata["Patch Size"]))
        # print(sub_image.shape, type(sub_image), " sub image")
        wrong_image = Image.fromarray(sub_image)
        wrong_image.save(Path(cwd  + "/training_sessions/" + timestamp + 
                "/incorrect_labelings/" + str(x) + "_" + str(int(correct_labels[x])) + ".tif"))

    # Plot training data

    plt.plot(data["training_losses"], label="train loss")
    plt.plot(data["validation_losses"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    fig_file = Path()
    loss_file  = Path(cwd  + "/training_sessions/" + timestamp + "/train_loss.png")
    plt.savefig(loss_file)
    plt.clf()
    
    plt.plot(data["training_errors"], label="train error")
    plt.plot(data["validation_errors"], label="val error")
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title("Training Error")
    plt.legend()
    fig_file = Path()
    error_file  = Path(cwd  + "/training_sessions/" + timestamp + "/train_error.png")
    plt.savefig(error_file)
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convolutional Neural Network(CNN) Model for 10-707(Deep Learning) project")
    parser.add_argument("arr_pos_root", help="File path to the positive data arrays. Excludes _(x/y).npy extension")
    parser.add_argument("arr_neg_root", help="File path to the negative data arrays. Excludes _(x/y).npy extension")

    arr_pos_root = sys.argv[1]
    arr_neg_root = sys.argv[2]

    x_pos, y_pos = import_data(arr_pos_root)
    x_neg, y_neg = import_data(arr_neg_root)

    x = torch.as_tensor(np.concatenate((x_pos, x_neg)), dtype=torch.float)
    y = torch.as_tensor(np.concatenate((y_pos, y_neg)), dtype=torch.long)
    y = y.view((len(y)))

    # y.cuda()
    # x.cuda()

    print("Shape of X: ", x.shape)
    print("Shape of y: ", y.shape)

    metadata_dict = {
        "Patch Size": x.shape[2],
        "Batch Size" : 42,
        "Pooling": 2,
        "Learning Rate" : 0.0001,
        "Kernel Size" : 3,
        "Padding" : 1,
        "Epochs" : 5,
        "Test Loss" : 0,
        "Test Error" : 0
    }

    # CAN START CROSS-VALIDATION LOOP HERE

    # Shuffle data before partitioning
    partition = (0.6, 0.3)
    x_training, x_val, x_test, y_training, y_val, y_test = rand_split_data(x, y, partition)

    print("Dataset Partitions:", str(len(x_training)) + " , " + str(len(x_val)) + " , " + str(len(x_test)) + " ")

    # Network Params: c1_out, c2_out, l1_out, l2_out, out_size, kernel_size, patch_size, pool_size

    c1_filters = 20
    c2_filters = 60

    f1_nodes = 500
    f2_nodes = 200

    net = Net(c1_filters, c2_filters, f1_nodes, f2_nodes, 2, metadata_dict["Kernel Size"], metadata_dict["Patch Size"], metadata_dict["Pooling"], metadata_dict["Padding"])
    net.batch_size = metadata_dict["Batch Size"]
    net.epochs = metadata_dict["Epochs"]
    net.float()
    # net.cuda()
    print(net)
    net.objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = metadata_dict["Learning Rate"])

    # Initial training metrics
    train_loss, train_error, _ = epoch_loss_error(net, x_training, y_training)
    print("\tTraining Loss: ", train_loss)
    print("\tTraining Error: ", pc(train_error))
    val_loss, val_error, _ = epoch_loss_error(net, x_val, y_val)
    print("\tValidation Loss: ", val_loss)
    print("\tValidation Error: ", pc(val_error))     
    loss_error = {
        "training_losses" : [train_loss],
        "validation_losses" : [val_loss],
        "training_errors" : [train_error],
        "validation_errors" : [val_error]
    }

    # Start Training
    for x in range(net.epochs):
        print("Epoch: ", x, " start")
        for y in range(int(len(x_training)/net.batch_size)):
    
            batch_x, batch_y = sample_batch(x_training, y_training, net.batch_size)
            optimizer.zero_grad()
            output = net(batch_x)
            loss = net.objective(output, batch_y)
            loss.backward()
            optimizer.step()

        train_loss, train_error, _ = epoch_loss_error(net, x_training, y_training)
        print("\tTraining Loss: ", train_loss)
        print("\tTraining Error: ", pc(train_error))
        val_loss, val_error, _ = epoch_loss_error(net, x_val, y_val)
        print("\tValidation Loss: ", val_loss)
        print("\tValidation Error: ", pc(val_error))    
    
        loss_error["training_losses"].append(train_loss)
        loss_error["training_errors"].append(train_error)
        loss_error["validation_losses"].append(val_loss)
        loss_error["validation_errors"].append(val_error)
        
    test_loss, test_error, wrong_labels = epoch_loss_error(net, x_test, y_test)

    print("Training Complete, Reporting Test Results")
    print("\tTest Loss: ", test_loss)
    print("\tTest Error: ", pc(test_error))
    metadata_dict["Test Loss"] = test_loss
    metadata_dict["Test Error"] = test_error

    record_training(loss_error, metadata_dict, net, x_test[wrong_labels], y_test[wrong_labels])





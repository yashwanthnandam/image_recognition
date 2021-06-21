import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np

from Net import Net, LeNet5
from dataset import GestureDataset
from utils import Tracker, get_predictions



def train(output_dir, batch_size=32, epochs=5, learning_rate=0.001, gpu=False):
    """Perform model training"""
        
    ##################################################
    # Making the dataset
    ##################################################
    transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    trainset = GestureDataset("train_phone.csv","../data1/", transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = GestureDataset("test_phone.csv","../data1/", transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('prev', 'next', 'pause', 'other')



    ##################################################
    #Showing some training images for fun:)
    ##################################################
    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



    ##################################################
    #Define the Net and Define Loss function and Optimiser
    ##################################################
    # net = LeNet5()
    net = LeNet5(ckpt="./ckpt/004.ckpt", gpu=False)
    net.eval()

    if gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


    ##################################################
    #Train the Network
    ##################################################
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Starting epoch {:03d}'.format(epoch))

        # statistic tracking
        train_loss_tracker = Tracker()
        train_accuracy_tracker = Tracker()

        for idx, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            mini_batch = inputs.size(0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds= get_predictions(outputs.squeeze().cpu().data.numpy())
            preds = np.array(preds) == labels.cpu().squeeze().data.numpy()
            accuracy = np.mean(preds)

            train_loss_tracker.step(loss.item() * mini_batch, mini_batch)
            train_accuracy_tracker.step(
                accuracy * mini_batch, mini_batch)


            if idx % 100 == 0:
                print('Batch {}, average loss {} - average accuracy {}'
                    .format(idx, train_loss_tracker.get_average(),
                            train_accuracy_tracker.get_average()))

        #do validation pass

        val_loss_tracker = Tracker()
        val_accuracy_tracker = Tracker()

        for data in testloader:
            inputs, labels = data
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            mini_batch = inputs.size(0)

            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                preds= get_predictions(outputs.squeeze().cpu().data.numpy())
                preds = np.array(preds) == labels.cpu().squeeze().data.numpy()
                accuracy = np.mean(preds)

                val_loss_tracker.step(loss.item() * mini_batch, mini_batch)
                val_accuracy_tracker.step(
                    accuracy * mini_batch, mini_batch)
        
        state_dict = {
            'n_classes': 3,
            'input_size': (50, 50),
            'state_dict': net.state_dict()
        }
        torch.save(state_dict, os.path.join(output_dir, '{:03d}.ckpt'.format(epoch)))
        print('Validation - loss {}, accuracy {}'.format(
            val_loss_tracker.get_average(),
            val_accuracy_tracker.get_average()
        ))


    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir',
                        help='The output directory to store checkpoint')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='The batch size')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Learing rate')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    print("go in main")
    train(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gpu=args.gpu)
import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms
import numpy as np
from models.model import Model
from tqdm import tqdm

batch_size = 1024
epochs = 1000
learning_rate = 0.00001


train_validation_dataset = ds.EMNIST(root = "../data", split = "letters", 
                                    train = True,
                                    transform = transforms.ToTensor(), 
                                    download = True)

test_dataset = ds.EMNIST(root = "../data", split = "letters",
                        train = False,
                        transform = transforms.ToTensor(),
                        download = True)


# split train and validation datasets 85-15
train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [int(0.85 * len(train_validation_dataset)), int(0.15 * len(train_validation_dataset))])

# keep only 2 batches for faster training
train_dataset = torch.utils.data.Subset(train_dataset, range(2 * batch_size))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

examples = enumerate(train_loader)
i, (samples, labels) = next(examples)
print(samples.numpy().reshape(-1, 28 * 28).shape, labels.numpy().shape)
print(labels)

# import matplotlib.pyplot as plt
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(train_dataset[i][0][0], cmap = 'gray', interpolation = 'none')
#     plt.title("Class {}".format(train_dataset[i][1]))

# plt.show()


# create model
from layers.dense_layer import DenseLayer
from layers.relu_layer import ReLU
from layers.softmax_layer import SoftmaxLayer
from layers.dropout_layer import DropoutLayer

network = [
    DenseLayer(28 * 28, 1024),
    ReLU(),
    DropoutLayer(0.3),
    DenseLayer(1024, 26)
]

from loss_functions.cross_entropy_loss import CrossEntropyLoss

loss = CrossEntropyLoss()

model = Model(network, loss)

# train model
for epoch in tqdm(range(epochs)):
    error = 0
    for i, (images, labels) in enumerate(train_loader):
        # convert to numpy
        images = images.numpy()
        labels = labels.numpy()
        images = images.reshape(-1, 28 * 28).T
        output = model.forward(images)
        # print(f"loss: {loss.f(output, labels)}")

        # one hot encode labels
        labels_one_hot = np.zeros((26, batch_size))
        labels_one_hot[labels - 1, np.arange(batch_size)] = 1

        error += model.loss.f(output, labels_one_hot)

        output_gradient = model.loss.df(output, labels_one_hot)
        model.backward(output_gradient, learning_rate)

    # check if loss is nan
    if np.isnan(model.loss.f(output, labels)):
        print(f"epoch: {epoch + 1}: Loss is nan")
        break
    if np.isinf(model.loss.f(output, labels)):
        print(f"epoch: {epoch + 1}: Loss is inf")
        break

    # error /= len(train_loader)
    # if epoch % 10 == 0:
    #     print(f"Epoch {epoch + 1} loss: {error}")

# test model
from sklearn.metrics import accuracy_score

n_correct = 0
n_samples = 0

for i, (images, labels) in enumerate(test_loader):
    images = images.numpy()
    labels = labels.numpy()
    images = images.reshape(-1, 28 * 28).T
    output = model.predict(images)
    output = np.argmax(output, axis=0)

    n_samples += labels.shape[0]
    n_correct += (output == labels - 1).sum().item()

acc = 100.0 * n_correct / n_samples
print(f"Accuracy: {acc}")
    


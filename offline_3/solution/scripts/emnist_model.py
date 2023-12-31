import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms
import numpy as np
from models.model import Model
from tqdm import tqdm
from layers.dense_layer import DenseLayer
from layers.relu_layer import ReLU
from layers.dropout_layer import DropoutLayer
from optimizers.adam import AdamOptimizer
from loss_functions.cross_entropy_loss import CrossEntropyLoss
from performance import report_scores, report_confusion_matrix
import pickle

model_name = "dense_relu_dropout_dense_005"
report_filename = "../reports/" + model_name + ".txt"
pickle_filename = "../pickles/" + model_name + ".pkl"
# inintialize hyperparameters
batch_size = 1024
epochs = 50
learning_rate = 0.005

# initialize output file
open(report_filename, "w").close()

train_validation_dataset = ds.EMNIST(root = "../data", split = "letters", 
                                    train = True,
                                    transform = transforms.ToTensor(), 
                                    download = True)


# split train and validation datasets 85-15
torch.manual_seed(7)
train_dataset, validation_dataset = torch.utils.data.random_split(train_validation_dataset, [int(0.85 * len(train_validation_dataset)), int(0.15 * len(train_validation_dataset))])

# take subset of train to test code
# train_dataset = torch.utils.data.Subset(train_dataset, range(0, batch_size * 2))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)


network = [
    DenseLayer(28 * 28, 1024),
    ReLU(),
    DropoutLayer(0.3),
    DenseLayer(1024, 26)
]

loss = CrossEntropyLoss()
model = Model(network, loss, name = model_name)
optimizer = AdamOptimizer(model.denselayers(), learning_rate = learning_rate)

# train model
for epoch in tqdm(range(epochs)):
    # error = 0
    for i, (images, labels) in enumerate(train_loader):
        # convert to numpy
        images = images.numpy()
        labels = labels.numpy()
        images = images.reshape(-1, 28 * 28).T
        output = model.forward(images)

        # one hot encode labels
        size = labels.shape[0]
        labels_one_hot = np.zeros((26, size))
        labels_one_hot[labels - 1, np.arange(size)] = 1

        # error += model.loss.f(output, labels_one_hot)

        output_gradient = model.loss.df(output, labels_one_hot)
        model.backward(output_gradient)

        optimizer.step()

    # report training loss, trainig accuracy, validation loss, validation accuracy and validation macro f1 score
    report_scores(model, train_loader, epoch + 1, loss,"Training", report_filename, False)
    report_scores(model, validation_loader, epoch + 1, loss,"Validation", report_filename, False)


report_confusion_matrix(model, train_loader, report_filename)

# save model using pickle
with open(pickle_filename, "wb") as f:
    pickle.dump(model, f)

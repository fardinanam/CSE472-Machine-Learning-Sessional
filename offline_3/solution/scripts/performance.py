import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models.model import Model
from loss_functions.loss_function import LossFunction
import seaborn as sns
import matplotlib.pyplot as plt

def prediction(model : Model, loader : torch.utils.data.DataLoader):
    images = []
    labels = []
    for batch in loader:
        images.append(batch[0])
        labels.append(batch[1])

    images = torch.cat(images).numpy()
    labels = torch.cat(labels).numpy()

    images = images.reshape(-1, 28 * 28).T
    output = model.predict(images)
    # output = np.argmax(output, axis = 0) + 1

    return output, labels

def report_scores(model : Model, loader : torch.utils.data.DataLoader, iteration : int, name : str, report_filename : str, show : bool = True):
    output, labels = prediction(model, loader)

    # one hot encode labels
    size = labels.shape[0]
    labels_one_hot = np.zeros((26, size))
    labels_one_hot[labels - 1, np.arange(size)] = 1

    loss = model.loss.f(output, labels_one_hot) / labels.shape[0]
    output = np.argmax(output, axis = 0) + 1
    # save report
    with open(report_filename, "a") as f:
        f.write(f"{name} iteration: {iteration}\n")
        f.write(f"{name} loss: {loss}\n")
        f.write(f"{name} accuracy: {accuracy_score(labels, output)}\n")
        f.write(f"{name} f1 score: {f1_score(labels, output, average = 'macro')}\n\n")
    
    if show:
        print(f"{name} iteration: {iteration}")
        print(f"{name} loss: {loss}")
        print(f"{name} accuracy: {accuracy_score(labels, output)}")
        print(f"{name} f1 score: {f1_score(labels, output, average = 'macro')}")

def report_confusion_matrix(model : Model, loader : torch.utils.data.DataLoader, report_filename : str):
    output, labels = prediction(model, loader)
    output = np.argmax(output, axis = 0) + 1

    # with open(report_filename, "a") as f:
    #     f.write(f"{confusion_matrix(labels, output)}\n\n")

    # use  seaborn to plot confusion matrix and save it
    # use minimum padding
    plt.figure(figsize = (14, 14))
    sns.heatmap(confusion_matrix(labels, output), annot = True, fmt = 'g', cmap = 'Blues', cbar = False)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.9)
    plt.savefig(report_filename.replace(".txt", ".png"))

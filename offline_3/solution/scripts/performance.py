import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models.model import Model
from loss_functions.loss_function import LossFunction

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
    output = np.argmax(output, axis = 0) + 1

    return output, labels

def report_scores(model : Model, loader : torch.utils.data.DataLoader, iteration : int , loss : LossFunction, name : str, report_filename : str, show : bool = True):
    output, labels = prediction(model, loader)

    # save report
    with open(report_filename, "a") as f:
        f.write(f"{name} iteration: {iteration}\n")
        f.write(f"{name} loss: {loss.f(output, labels)}\n")
        f.write(f"{name} accuracy: {accuracy_score(labels, output)}\n")
        f.write(f"{name} f1 score: {f1_score(labels, output, average = 'macro')}\n\n")
    
    if show:
        print(f"{name} iteration: {iteration}")
        print(f"{name} loss: {loss.f(output, labels)}")
        print(f"{name} accuracy: {accuracy_score(labels, output)}")
        print(f"{name} f1 score: {f1_score(labels, output, average = 'macro')}")

def report_confusion_matrix(model : Model, loader : torch.utils.data.DataLoader, report_filename : str):
    output, labels = prediction(model, loader)
    
    with open(report_filename, "a") as f:
        f.write(f"{confusion_matrix(labels, output)}\n\n")
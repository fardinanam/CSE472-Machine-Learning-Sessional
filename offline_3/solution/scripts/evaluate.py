import pickle
import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch
import numpy as np
from performance import prediction, report_scores, report_confusion_matrix

pickle_filename = "../pickles/dense_relu_dropout_dense_005.pkl"
report_filename = "../reports/emnist_model_report.txt"

open(report_filename, "w").close()

with open(pickle_filename, "rb") as f:
    model = pickle.load(f)

test_dataset = ds.EMNIST(root = "../data", split = "letters",
                        train = False,
                        transform = transforms.ToTensor(),
                        download = True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

report_scores(model, test_loader, 0, model.loss, "test", report_filename)
report_confusion_matrix(model, test_loader, report_filename)
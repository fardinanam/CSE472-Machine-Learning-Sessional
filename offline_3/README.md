## Feed Forward Neural Network

### Introduction
A modular implementation of a feed forward neural network in Python. The code is written in a modular fashion, so that it can be easily extended to include new activation functions, cost functions, and network architectures.

Initial testing is done on the EMNIST dataset.

### Training on EMNIST Dataset

Run the following command to train the model on the EMNIST dataset.

```sh
    python emnist_model.py
```

The model will be trained for 15 epochs. The training and validation accuracy will be printed after each epoch. The final test accuracy will be printed after the training is complete.

### Testing on EMNIST Dataset

Run the following command to test the model on the EMNIST dataset.

```sh
    python evaluate.py
```

The test accuracy and macro f1 score will be printed.
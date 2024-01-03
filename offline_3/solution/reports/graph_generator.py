import matplotlib.pyplot as plt
import re
from tqdm import tqdm

reports = [
    
    'dense_relu_dropout_dense_512_001.txt',
    'dense_relu_dropout_dense_512_002.txt',
    'dense_relu_dropout_dense_512_0035.txt',
    'dense_relu_dropout_dense_512_005.txt',
    'dense_relu_dropout_dense_001.txt',
    'dense_relu_dropout_dense_002.txt',
    'dense_relu_dropout_dense_0035.txt',
    'dense_relu_dropout_dense_005.txt',
    'dense_sigmoid_dropout_dense_001.txt',
    'dense_sigmoid_dropout_dense_002.txt',
    'dense_sigmoid_dropout_dense_0035.txt',
    'dense_sigmoid_dropout_dense_005.txt'
]

graphs_dir = "graphs/"

for report in tqdm(reports):
# Open the text file and read its contents
    with open(report, 'r') as f:
        lines = f.readlines()

    # Prepare lists to hold the data
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    epochs = []

    # Define a regular expression pattern to match the lines with the metrics
    pattern = r'(\w+) (iteration|loss|accuracy|f1 score): ([\d\.]+)'

    # Iterate over the lines in the file
    for line in lines:
        match = re.match(pattern, line)
        if match:
            metric_type, metric_name, metric_value = match.groups()
            if metric_type == 'Training' and metric_name == 'iteration':
                epochs.append(int(metric_value))
            elif metric_name == 'loss':
                if metric_type == 'Training':
                    training_loss.append(float(metric_value))
                else:
                    validation_loss.append(float(metric_value))
            elif metric_name == 'accuracy':
                if metric_type == 'Training':
                    training_accuracy.append(float(metric_value))
                else:
                    validation_accuracy.append(float(metric_value))

    # Create a new figure
    fig = plt.figure(figsize=(16, 6))

    # Create the first subplot for loss vs epoch
    plt.subplot(1, 2, 1)  # 2 rows, 1 column, first plot
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Create the second subplot for accuracy vs epoch
    plt.subplot(1, 2, 2)  # 2 rows, 1 column, second plot
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the figure
    plt.savefig(graphs_dir + report[:-4] + '.png')
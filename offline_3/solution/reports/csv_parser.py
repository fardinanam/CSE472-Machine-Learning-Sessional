import csv
import re

# Open the text file and read its contents
with open('dense_relu_dropout_dense_002.txt', 'r') as f:
    lines = f.readlines()

# Prepare a list to hold the data
data = []

# Define a regular expression pattern to match the lines with the metrics
pattern = r'(\w+) (iteration|loss|accuracy|f1 score): ([\d\.]+)'

# Iterate over the lines in the file
for line in lines:
    match = re.match(pattern, line)
    if match:
        metric_type, metric_name, metric_value = match.groups()
        if metric_type == 'Training' and metric_name == 'iteration':
            # Start a new row for each training iteration
            data.append({})
        if data:
            # Add the metric to the current row
            data[-1][f'{metric_type} {metric_name}'] = float(metric_value)

# Write the data to a CSV file
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
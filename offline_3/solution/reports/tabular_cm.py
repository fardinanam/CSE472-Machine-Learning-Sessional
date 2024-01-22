import numpy as np
import pandas as pd

# Your confusion matrix as a string
confusion_matrix_str = """
[[712   5   3   6   3   0   6   3   0   0   4   0   0   6   4   3  33   1
    0   3   3   0   0   1   0   4]
 [  2 757   0   2   2   0   4   9   1   1   1   4   0   1   7   0   4   1
    0   0   1   0   0   0   0   3]
 [  0   0 763   1  15   0   1   0   0   0   0   6   0   0   5   0   1   2
    1   1   2   0   1   0   0   1]
 [  4   6   1 720   0   0   0   1   0   6   1   0   0   3  44   3   3   2
    2   1   1   0   1   0   1   0]
 [  3   3  16   0 757   0   4   0   1   0   2   2   0   0   1   2   3   1
    0   3   1   0   0   0   0   1]
 [  0   0   0   1   3 727   6   0   1   0   0   2   0   0   0  20   2   5
    3  28   0   0   0   0   1   1]
 [ 12  16   9   1   1   3 589   1   0   5   0   1   0   1   2   0 149   2
    5   0   0   0   1   0   1   1]
 [  4   6   0   3   0   0   0 735   1   1   8   8   8  18   0   0   1   2
    0   0   2   0   2   1   0   0]
 [  0   0   1   0   1   1   1   1 663  15   1 104   0   0   1   0   0   0
    2   1   1   0   1   1   0   5]
 [  1   2   0   6   0   0   5   0  24 745   0   1   0   0   0   0   1   0
    3   9   2   0   0   0   0   1]
 [  1   3   1   1   1   0   0  13   3   1 747   3   0   1   0   0   0   4
    0   4   0   2   0  14   1   0]
 [  0   1   4   0   0   0   0   6 264   1   0 518   0   0   1   0   1   1
    0   1   0   0   0   0   2   0]
 [  1   0   0   1   0   0   0   2   0   0   1   0 783   6   0   0   0   1
    0   0   1   0   3   0   1   0]
 [ 10   1   0   6   0   0   0   8   0   1   3   1   9 741   1   0   1   2
    0   1   2   3   7   2   0   1]
 [  1   0   3   3   0   0   1   0   0   0   0   0   0   0 786   0   3   1
    0   0   2   0   0   0   0   0]
 [  0   0   0   6   1   3   1   0   0   0   0   0   0   1   0 766   9   7
    0   5   0   0   0   0   1   0]
 [ 17   1   2   1   4   1  48   0   5   3   0   0   0   0   5   2 701   4
    0   1   1   0   0   1   3   0]
 [  5   1   1   0   4   3   1   0   1   0   7   1   1   1   0   2   2 743
    1   8   0   8   0   3   2   5]
 [  4   2   0   1   1   1  12   1   0  10   0   0   0   1   1   0   2   0
  763   0   0   0   0   0   0   1]
 [  1   3   2   1   2   3   0   1   2   2   5   2   0   0   1   0   0   1
    0 765   0   0   0   2   6   1]
 [  0   0   1   3   0   0   1   2   0   2   3   1   3   2   2   0   3   0
    1   0 750  18   4   0   4   0]
 [  0   0   0   3   0   0   0   0   1   1   0   0   0   0   0   0   0   6
    0   2  41 728   2   3  13   0]
 [  0   0   0   2   1   0   0   2   0   0   1   0   3   5   0   0   2   0
    0   1  10   1 772   0   0   0]
 [  2   0   0   0   0   0   0   2   1   1  12   1   0   1   0   0   3   2
    0   1   0   3   0 759   9   3]
 [  0   0   0   2   0   0   5   2   0   7   0   2   0   0   0   1   5   4
    1   3   0  10   0   7 751   0]
 [  0   2   0   1   3   0   2   2   7   0   0   1   0   0   0   0   2   2
    0   2   0   0   0   2   0 774]]
"""

# Remove the outer brackets and split the string into lines
lines = confusion_matrix_str.strip()[1:-1].split("\n")

# Parse each line into a list of integers
confusion_matrix = [list(map(int, line.strip()[1:-1].split())) for line in lines]

# Convert the list of lists into a numpy array
confusion_matrix = np.array(confusion_matrix)

# Create a DataFrame from the numpy array
df = pd.DataFrame(confusion_matrix)

# Save the DataFrame as a CSV file
df.to_csv("confusion_matrix.csv", index=False, header=False)
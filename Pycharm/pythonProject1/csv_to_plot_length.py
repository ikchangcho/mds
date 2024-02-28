import re  # For regular expression operations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing your CSV files
directory = '/Users/ik/Pycharm/pythonProject1'

# Desired number of rows and value of L
desired_L = 30
initial_dim = 3
final_dim = int(2 * desired_L) - initial_dim + 1

# Initialize a figure for plotting
plt.figure(figsize=(10, 6))

# Regular expression pattern to extract L, N1, and N2 from filenames
pattern = r'length(\d+)_(\d+)\+(\d+)\.csv'

# List to store data and labels
data_list = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    match = re.match(pattern, filename)
    if match:
        L_value, N1_value, N2_value = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if L_value == desired_L:
            # Construct the full file path
            filepath = os.path.join(directory, filename)

            # Read the CSV file
            data = pd.read_csv(filepath, header=None, skiprows=initial_dim - 1)  # Assuming there's no header row

            # Adjust the number of rows
            if len(data) > final_dim:
                # Truncate if there are more rows than needed
                data = data.iloc[:final_dim]
            elif len(data) < final_dim:
                # Pad with zeros if there are fewer rows than needed
                pad_length = final_dim - len(data)
                padding = pd.DataFrame([[0] * pad_length])  # Ensuring the padding has the correct shape
                data = pd.concat([data, padding], ignore_index=True, axis=0)

            # Store data and sum of N1 and N2 as a tuple
            data_list.append((N1_value, N2_value, data.iloc[:, 0]))

# Sort the list based on the sum of N1 and N2 (first element of the tuple)
data_list.sort(key=lambda x: x[0])

# Initialize a figure for plotting
plt.figure(figsize=(10, 6))

import matplotlib.ticker as ticker

# Plot the sorted data
for N1, N2, data in data_list:
    # Create an x-axis that starts from initial_dim and has the same length as the data
    x_axis = np.arange(initial_dim, initial_dim + len(data))

    # Plot the data with the custom x-axis
    plt.plot(x_axis, data, 'o-', label=f'{N1}+{N2}')

# Customize the plot
plt.title(f'Model 2: Longest common subsequence\nLength {desired_L}')
plt.xlabel('Dimension')
plt.ylabel('Normalized Stress')
plt.legend()

# Ensure that only integer values are shown on the x-axis
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.savefig(f'compare_length{desired_L}.png')
plt.show()

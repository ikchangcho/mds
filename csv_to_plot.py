import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import MaxNLocator  # Import MaxNLocator for integer x-axis


def read_and_adjust_csv(file_path, initial_row, final_row):
    # Read specified rows of the CSV file
    df = pd.read_csv(file_path, header=None, skiprows=initial_row - 1, nrows=final_row - initial_row + 1)
    return df


folder_path = '/Users/ik/Pycharm/pythonProject1/csvToPlot'  # Replace with your folder path
initial_row = 1  # Start plotting from this row
final_row = 20  # End plotting at this row

file_paths = glob.glob(f'{folder_path}/*.csv')

if not file_paths:
    print("No CSV files found in the specified folder.")

# Create a single figure and axes
plt.figure(figsize=(10, 5))

lines = []
for file_path in file_paths:
    df = read_and_adjust_csv(file_path, initial_row, final_row)

    # Plot each CSV file's data on the same Axes
    line, = plt.plot(range(initial_row, final_row + 1), df.iloc[:, 0], '-o')  # Plot and unpack the line object
    lines.append(line)  # Store the line object
    #plt.plot(range(initial_row, final_row + 1), df.iloc[:, 0], label=file_path)

plt.title('Length 10')
plt.xlabel('Dimension')
plt.ylabel('Stress')

labels = ['Simplest', 'Longest common sequence']
for line, label in zip(lines, labels):
    line.set_label(label)
plt.legend(loc='best')  # Add a legend to distinguish different CSV files
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis ticks are integers
plt.tight_layout()
plt.savefig(f'two_models_length10.png')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import MaxNLocator  # Import MaxNLocator for integer x-axis


def read_and_adjust_csv(file_path, initial_row, final_row):
    # Read specified rows of the CSV file
    df = pd.read_csv(file_path, header=None, skiprows=initial_row - 1, nrows=final_row - initial_row + 1)
    return df


folder_path = '/Users/ik/Pycharm/pythonProject1/csvToPlot'  # Replace with your folder path
initial_row = 3  # Start plotting from this row
final_row = 60  # End plotting at this row

file_path = glob.glob(f'{folder_path}/*.csv')

if not file_path:
    print("No CSV files found in the specified folder.")

# Create a single figure and axes
plt.figure(figsize=(10, 5))

lines = []
dfs = []
for file_name in file_path:
    df, = read_and_adjust_csv(file_name, initial_row, final_row)
    dfs.append(df)

print(df)
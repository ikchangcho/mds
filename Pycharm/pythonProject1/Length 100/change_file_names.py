import os
import re

# Define the directory containing the files
directory = '/Users/ik/Pycharm/pythonProject1'

# Regular expression to match the current filename pattern and extract N1, N2, L
pattern = re.compile(r'length(\d+)N1(\d+)N2(\d+)')

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the filename matches the pattern
    match = pattern.match(filename)
    if match:
        # Extract N1, N2, L from the filename
        L, N1, N2 = match.groups()

        # Construct the new filename
        new_filename = f'length{L}_{N1}+{N2}'

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

        print(f'Renamed "{filename}" to "{new_filename}"')

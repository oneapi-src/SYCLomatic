#!/root/miniconda3/bin/python

import os
import csv
import sys

def generate_array_declaration(csv_file):
    filename = os.path.splitext(os.path.basename(csv_file))[0]  # Extract filename without extension
    array_name = filename.replace(" ", "_")  # Replace spaces in filename with underscores
    array_declaration = f"{array_name} = ["

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header

        for row in csv_reader:
            if len(row) >= 2 and row[1] == "NO":
                array_declaration += f"'{row[0]}', "

    array_declaration = array_declaration.rstrip(", ")  # Remove trailing comma and space
    array_declaration += "]"

    return array_declaration

if __name__ == '__main__':
    if '-p' in sys.argv:
        folder_path = sys.argv[sys.argv.index('-p') + 1]
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            array_declaration = generate_array_declaration(csv_path)
            print(array_declaration)
    else:
        print("Please specify the folder path using the -p parameter. The default folder stores .csv files <should be under SYCLomatic_repository/docs/dev_guide/api-mapping-status>")


import os
import re


def parse_filename(filename):
    pattern = r"\d+\.\d+|\d+"
    numbers = re.findall(pattern, filename)
    converted_numbers = []
    for number in numbers:
        try:
            converted = int(number)
        except ValueError:
            converted = float(number)
        converted_numbers.append(converted)
    return tuple(converted_numbers)


def parse_directory(directory_path):
    files = os.listdir(directory_path)
    parsed_files = {}
    parsed_files_list = []
    for file in files:
        if file.endswith(".png"):
            tup = parse_filename(file)
            parsed_files[file] = tup
            parsed_files_list.append(tup)
    return parsed_files, parsed_files_list


# directory_path = "grey"
# parsed_data, parsed_files_list = parse_directory(directory_path)
# print(parsed_data)
# print("---")
# print(parsed_files_list)

grey = [
    (16, 3, 0.3, 0.0001, 64, 3000),
    (16, 5, 0.2, 0.01, 128, 500),
    (16, 7, 0.4, 0.0001, 64, 1000),
    (16, 7, 0.2, 0.0001, 128, 1000),
    (16, 3, 0.3, 0.005, 64, 500),
    (16, 5, 0.5, 0.001, 64, 2000),
    (16, 7, 0.4, 0.001, 128, 3000),
    (16, 7, 0.4, 0.0001, 32, 2000),
    (16, 4, 0.1, 0.001, 128, 500),
]

purple = [
    (16, 6, 0.4, 0.0005, 128, 3000),
    (16, 4, 0.4, 0.0005, 128, 500),
    (16, 6, 0.2, 0.0005, 16, 1000),
    (16, 4, 0.4, 0.0005, 64, 1000),
    (16, 3, 0.2, 0.001, 16, 1000),
    (16, 6, 0.2, 0.001, 32, 1000),
    (16, 8, 0.4, 0.001, 16, 2000),
    (16, 6, 0.1, 0.001, 128, 1000),
    (16, 4, 0.4, 0.01, 128, 1000),
    (16, 6, 0.5, 0.0005, 128, 3000),
    (16, 7, 0.5, 0.0001, 64, 2000),
    (16, 8, 0.3, 0.001, 32, 2000),
    (16, 8, 0.2, 0.001, 32, 500),
    (16, 4, 0.1, 0.001, 64, 500),
    (16, 6, 0.1, 0.0005, 128, 2000),
    (16, 3, 0.1, 0.001, 16, 3000),
    (16, 4, 0.4, 0.01, 16, 2000),
    (16, 4, 0.2, 0.001, 128, 3000),
    (16, 4, 0.1, 0.0005, 128, 2000),
    (16, 4, 0.2, 0.001, 64, 2000),
    (16, 6, 0.3, 0.0005, 128, 1000),
    (32, 4, 0.3, 0.01, 16, 3000),
    (16, 6, 0.4, 0.001, 32, 500),
    (16, 6, 0.2, 0.001, 32, 3000),
    (16, 4, 0.5, 0.001, 64, 500),
    (32, 4, 0.3, 0.01, 128, 1000),
    (16, 4, 0.3, 0.001, 64, 2000),
    (16, 8, 0.1, 0.0001, 16, 1000),
    (16, 8, 0.1, 0.01, 16, 1000),
    (16, 3, 0.1, 0.01, 64, 1000),
    (16, 5, 0.1, 0.0005, 32, 3000),
    (16, 4, 0.2, 0.001, 16, 1000),
    (16, 6, 0.4, 0.001, 16, 500),
    (16, 7, 0.3, 0.01, 16, 3000),
    (16, 5, 0.1, 0.001, 64, 500),
    (16, 7, 0.4, 0.001, 64, 1000),
    (16, 3, 0.3, 0.005, 128, 500),
    (16, 8, 0.3, 0.0001, 64, 500),
    (16, 6, 0.5, 0.01, 32, 2000),
    (16, 4, 0.4, 0.001, 64, 3000),
    (16, 7, 0.1, 0.0001, 64, 2000),
    (16, 7, 0.3, 0.0001, 32, 3000),
    (16, 4, 0.2, 0.0005, 16, 500),
    (16, 7, 0.5, 0.0001, 16, 1000),
    (16, 6, 0.3, 0.0005, 64, 3000),
    (16, 4, 0.5, 0.0005, 16, 3000),
    (16, 3, 0.5, 0.01, 32, 3000),
    (16, 7, 0.3, 0.0001, 32, 1000),
    (16, 8, 0.2, 0.001, 16, 500),
    (16, 8, 0.4, 0.0001, 16, 500),
    (16, 4, 0.3, 0.0005, 32, 3000),
    (16, 4, 0.1, 0.0005, 64, 2000),
    (16, 5, 0.2, 0.001, 32, 500),
    (16, 3, 0.1, 0.0001, 16, 3000),
    (16, 3, 0.1, 0.01, 32, 1000),
    (16, 4, 0.3, 0.01, 64, 2000),
    (16, 6, 0.4, 0.01, 64, 3000),
    (16, 7, 0.1, 0.01, 16, 1000),
    (16, 5, 0.2, 0.001, 16, 2000),
    (16, 3, 0.1, 0.01, 64, 3000),
    (16, 7, 0.3, 0.001, 16, 1000),
    (16, 4, 0.1, 0.001, 16, 1000),
    (16, 6, 0.2, 0.01, 128, 500),
    (16, 3, 0.5, 0.0005, 64, 3000),
    (16, 7, 0.4, 0.0001, 64, 500),
    (16, 5, 0.1, 0.01, 64, 1000),
    (16, 3, 0.1, 0.001, 16, 2000),
    (16, 7, 0.1, 0.0001, 16, 2000),
    (16, 8, 0.3, 0.001, 128, 500),
    (16, 7, 0.3, 0.01, 16, 500),
    (16, 5, 0.1, 0.0005, 16, 1000),
    (16, 7, 0.1, 0.0001, 16, 500),
    (16, 6, 0.5, 0.0005, 32, 1000),
    (16, 7, 0.3, 0.0001, 16, 1000),
    (16, 8, 0.4, 0.0001, 128, 1000),
    (16, 6, 0.4, 0.01, 128, 3000),
    (16, 3, 0.3, 0.0005, 128, 2000),
    (16, 4, 0.2, 0.001, 64, 1000),
    (16, 7, 0.5, 0.01, 128, 2000),
    (16, 6, 0.5, 0.0005, 64, 3000),
    (16, 6, 0.2, 0.001, 128, 2000),
    (16, 8, 0.5, 0.0001, 32, 1000),
    (16, 8, 0.1, 0.01, 16, 2000),
    (16, 3, 0.3, 0.0005, 64, 3000),
    (16, 7, 0.3, 0.001, 64, 3000),
    (32, 4, 0.2, 0.01, 16, 1000),
    (16, 7, 0.5, 0.0001, 64, 500),
    (16, 8, 0.3, 0.001, 32, 500),
    (16, 8, 0.5, 0.0001, 64, 3000),
    (16, 4, 0.3, 0.001, 32, 3000),
    (16, 8, 0.2, 0.001, 128, 3000),
    (16, 4, 0.4, 0.0005, 128, 2000),
    (16, 8, 0.2, 0.01, 16, 1000),
    (16, 8, 0.4, 0.01, 16, 500),
    (16, 3, 0.3, 0.01, 128, 1000),
    (16, 6, 0.2, 0.001, 32, 2000),
    (32, 4, 0.3, 0.01, 32, 500),
    (16, 5, 0.5, 0.01, 128, 500),
    (16, 7, 0.5, 0.001, 16, 3000),
    (16, 7, 0.3, 0.001, 32, 3000),
    (16, 4, 0.1, 0.001, 32, 3000),
    (16, 6, 0.5, 0.001, 16, 2000),
    (16, 7, 0.4, 0.001, 16, 3000),
    (16, 8, 0.1, 0.001, 128, 500),
    (16, 6, 0.1, 0.0005, 128, 1000),
    (16, 4, 0.4, 0.01, 16, 1000),
    (16, 7, 0.5, 0.0001, 32, 3000),
    (16, 6, 0.3, 0.0005, 16, 3000),
    (16, 4, 0.2, 0.01, 16, 3000),
    (16, 7, 0.1, 0.001, 16, 500),
    (16, 8, 0.2, 0.0001, 128, 3000),
    (16, 7, 0.2, 0.001, 128, 500),
    (16, 3, 0.2, 0.001, 32, 500),
    (16, 4, 0.1, 0.0005, 64, 1000),
    (32, 4, 0.2, 0.0005, 16, 1000),
    (16, 6, 0.1, 0.0005, 16, 500),
    (16, 3, 0.1, 0.001, 128, 3000),
    (16, 5, 0.5, 0.0005, 16, 1000),
    (16, 6, 0.5, 0.01, 64, 1000),
    (16, 6, 0.4, 0.001, 128, 1000),
    (16, 3, 0.1, 0.001, 32, 1000),
    (16, 7, 0.3, 0.01, 16, 2000),
    (16, 7, 0.5, 0.01, 16, 500),
    (16, 3, 0.4, 0.01, 32, 500),
    (16, 8, 0.1, 0.01, 128, 1000),
    (32, 4, 0.2, 0.001, 128, 3000),
    (32, 4, 0.3, 0.001, 16, 500),
    (16, 4, 0.2, 0.01, 128, 3000),
    (16, 6, 0.4, 0.0005, 16, 3000),
    (16, 7, 0.1, 0.01, 16, 500),
    (16, 4, 0.5, 0.001, 64, 2000),
    (16, 5, 0.4, 0.001, 64, 3000),
    (16, 6, 0.1, 0.01, 128, 500),
    (16, 6, 0.4, 0.01, 32, 500),
    (16, 4, 0.2, 0.0005, 32, 3000),
    (16, 5, 0.1, 0.0005, 128, 2000),
    (16, 7, 0.4, 0.0001, 16, 1000),
    (16, 6, 0.4, 0.01, 64, 2000),
    (16, 6, 0.2, 0.01, 128, 2000),
    (16, 8, 0.5, 0.001, 32, 2000),
    (16, 8, 0.1, 0.001, 16, 500),
    (16, 5, 0.3, 0.01, 16, 500),
    (16, 4, 0.3, 0.0005, 32, 2000),
    (16, 7, 0.1, 0.01, 128, 1000),
    (16, 5, 0.5, 0.001, 128, 500),
    (16, 7, 0.1, 0.0001, 32, 500),
    (16, 3, 0.2, 0.0001, 128, 1000),
]

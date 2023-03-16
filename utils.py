import csv

def load_params(path):
    """
    Function to load in the parameters for each experiment
    """
    config = {}
    with open(path) as file:
        config_file = csv.reader(file, delimiter=',')
        for line in config_file:
            if '.' in line[1] or 'e' in line[1]:
                config[line[0]] = float(line[1])
            else:
                config[line[0]] = int(line[1])
    return config
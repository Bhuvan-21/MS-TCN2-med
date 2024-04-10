import os 
import numpy as np


def write_str_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

        
def load_action_map(mapping_file):
    actions = read_file(mapping_file).split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    
    return actions_dict

        
def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

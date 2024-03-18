import os 
import numpy as np


def write_str_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

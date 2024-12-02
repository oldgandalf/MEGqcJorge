import os
import sys

path_to_entrypoint= os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir))

print(path_to_entrypoint)
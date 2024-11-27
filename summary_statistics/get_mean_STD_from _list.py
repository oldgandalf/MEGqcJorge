import ancpbids
import os 
import sys
import subprocess
import pandas as pd

std_tables = pd.read_csv('PSDtable_list.csv', index_col=False)
print(std_tables)

test = std_tables.values.tolist()
stringlist = [''.join(ele) for ele in test]
print(type(stringlist))
print(stringlist)
for elem in stringlist:
    print(elem)
    df=pd.read_csv(elem, sep='\t')
    print(df[['STD all']].mean())
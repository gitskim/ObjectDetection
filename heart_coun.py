import os
import sys
import numpy as np

filename = sys.argv[1]

mdict = dict()
with open(filename) as mfile:
    for line in mfile:
        if line.startswith('[') or line.endswith(']\n') or line.endswith(']'):
            continue
        print(line)
        resultarr = line.split(' ')
        print(resultarr)
        result = resultarr[1].split(':')[1]
        result = result.strip('\'')
        if result not in mdict:
            mdict.update({result: 1})
        else:
            mdict[result] += 1

maxval = 0
maxkey = ''
for k, v in mdict.items():
    if maxval < v:
        maxval = v
        maxkey = k

print(mdict)

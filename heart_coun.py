import sys

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

orted_x = sorted([(value,key) for (key,value) in mdict.items()], reverse=True)
for i in range(10):
    print(orted_x[i])

# print(orted_x)
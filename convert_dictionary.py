import ast
import json
import numpy as np
import re
import pickle
from ast import literal_eval
import ast
from pprint import pprint
# story/.MXF/.png
filename = "suhyun-object-test-11-13.txt"

dtype0 = "dtype=uint8)"
dtype = ", dtype=uint8)"
dtype2 = ", dtype=float32)"
dtype3 = "dtype=float32),"
array = "array("
dvvm = "{\'/dvmm"
dvvm_without = "\"/dvmm"
local_string_dict = ""
flag = False
converted = None
detection_classes = "detection_classes"

# print(pickle.load(filename))


# can convert one dictionary
# with open(filename, 'r') as opened_file:
#     newline = opened_file.read()
#     newline = newline.replace(dtype, "")
#     newline = newline.replace(dtype2, "")
#     newline = newline.replace(dtype3, "")
#     newline = newline.replace(array, "")
#     newline = newline.replace('\'', '\"')
#     result = ast.literal_eval(newline)
#     print(result)

# attempt to convert multiple dictionaries
# with open(filename, 'r') as opened_file:
#     newline = opened_file.read()
#     newline = newline.replace(dtype0, "")
#     newline = newline.replace(dtype, "")
#     newline = newline.replace(dtype2, "")
#     newline = newline.replace(dtype3, "")
#     newline = newline.replace(array, "")
#     newline = newline.replace(dvvm, dvvm_without)
#     newline = newline.replace('\'', '\"')
#     newline = newline.replace(dvvm_without, dvvm, 1)
#     result = ast.literal_eval(newline)
#     print(result)

# to convert multiple dictionareis
counter = 0
mlist = []
newline = ""

mread = 1
seen1 = False
seen2 = False
# with open(filename, 'r') as opened_file:
#     for counter in range(0,3):
#         while not seen1 and not seen2:
#             mread = opened_file.read(1)
#             newline += str(mread)
#             if seen1 == False and mread == '}':
#                 seen1 = True
#             if seen2 == False and mread == '}':
#                 seen2 = True
#
#             print(newline)
#             newline = newline.replace(dtype, "")
#             newline = newline.replace(dtype2, "")
#             newline = newline.replace(dtype3, "")
#             newline = newline.replace(array, "")
#             newline = newline.replace('\'', '\"')
#         if seen1 == True and seen2 == True:
#             result = ast.literal_eval(newline)
#             print(result)
#             mlist.append(result)
#             seen1 = False
#             seen2 = False


# with open(filename, 'r') as opened_file:
#     while mread:
#         while 1:
#             if counter == 2:
#                 break
#             mread = opened_file.read(1)
#             newline += str(mread)
#             if mread == '}':
#                 counter += 1
#
#         newline = newline.replace(dtype0, "")
#         newline = newline.replace(dtype, "")
#         newline = newline.replace(dtype2, "")
#         newline = newline.replace(dtype3, "")
#         newline = newline.replace(array, "")
#         newline = newline.replace('\'', '\"')
#         result = ast.literal_eval(newline)
#         print(result)
#         mlist.append(result)
#         counter = 0
#     print("------exception---------------------------")
#     print(mlist)


lines = ""
flag = False
dictionary = {}
with open(filename, 'r') as opened_file:
    while 1:
        newline = opened_file.readline()
        if not newline:
            break
        newline = newline.replace(dtype0, "")
        newline = newline.replace(dtype, "")
        newline = newline.replace(dtype2, "")
        newline = newline.replace(dtype3, "")
        newline = newline.replace(array, "")
        newline = newline.replace('\'', '\"')
        # print(newline)
        if "}}" in newline:
            flag = True
        if flag:
            lines += newline
            result = ast.literal_eval(lines)
            # mlist.append(result)
            # print(result)
            for key, val in result.items():
                #name
                namelist = key.split('/')
                namelist2 = [""] * 3

                for item in namelist:
                    if not item:
                        continue

                    if item.endswith(".MXF") or item.endswith(".MP4") or item.endswith(".mp4"):
                        namelist2[1] = item

                    elif item.endswith(".png"):
                        namelist2[2] = item

                    else:
                        namelist2[0] = namelist2[0] + '/' + item

                first = namelist2[0]
                second = namelist2[1]
                third = namelist2[2]
                fourth = val
                # d = {namelist2[0]: {namelist2[1]: {namelist2[2]: val}}}

                if first in dictionary:
                    if second in dictionary[first]:
                        if third in dictionary[first][second]:
                            dictionary[first][second][third].append(fourth)
                        else:
                            dictionary[first][second][third] = [fourth]
                    else:
                        dictionary[first][second] = {third: [fourth]}
                else:
                    dictionary[first] = {second: {third: [fourth]}}

                # dictionary.update(d)
                # print(dictionary)
            lines = ""
            flag = False
        else:
            lines += newline

print(dictionary)

result_file_name = "oid_11_13.file"
with open(result_file_name, 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(result_file_name, 'rb') as handle:
    b = pickle.load(handle)

print(dictionary == b)
# print(mlist)
print(type(mlist[0]))
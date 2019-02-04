import codecs
import os

# BLOCKSIZE = 1048576 # or some other, desired size in bytes
# directory_in_str = 'test_newspaper'
# directory = os.fsencode(directory_in_str)
# counter = 0
# for file in os.listdir(directory):
#     counter = counter + 1
#     filename = os.fsdecode(file)
#     if filename.endswith(".txt"):
#         sourceFileName = directory_in_str + '/' + filename
#         targetFileName = directory_in_str + '/new/' + filename
#         print('***\t', counter, '\t***')
#         with codecs.open(sourceFileName, 'r', 'Windows-1252', 'ignore') as sourceFile:
#             with codecs.open(targetFileName, 'w', 'Windows-1256', 'ignore') as targetFile:
#                 while True:
#                     contents = sourceFile.read(BLOCKSIZE)
#                     if not contents:
#                         break
#                     targetFile.write(contents)
#         continue
#     else:
#         continue


def read_input(input_file):
    with open(input_file, encoding='cp1252', errors='replace') as f:
        content = f.readlines()
        return content


print("ÏÇäÔÇååÇ".encode('cp1252', errors='ignore').decode('cp1256'))


import csv

csv_file_name = 'data/sts_benchmark/sts-train.csv'
test_file_name = 'data/sts_benchmark/sts-test.csv'
sentence1_file_name = 'data/sts_benchmark/sent1_test.txt'
sentence2_file_name = 'data/sts_benchmark/sent2_test.txt'
scores_file_name = 'data/sts_benchmark/scores_test.txt'

# with open(csv_file_name) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter='\t')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 2000:
#             break
#         with open(scores_file_name, 'a') as file:
#             file.write(row[4] + '\n')
#         with open(tr1_file_name, 'a') as file:
#             file.write(row[5] + '\n')
#         with open(tr2_file_name, 'a') as file:
#             file.write(row[6] + '\n')
#         line_count += 1

import pandas as pnd

dataframe = pnd.read_csv(test_file_name, '\t', quoting=3, error_bad_lines=False)
scores = dataframe['2.500'].tolist()
sentence1 = dataframe['A girl is styling her hair.'].tolist()
sentence2 = dataframe['A girl is brushing her hair.'].tolist()


def write_on_file(lst, file_name):
    with open(file_name, 'w') as file:
        for item in lst:
            file.write(str(item))
            file.write('\n')


write_on_file(scores, scores_file_name)
write_on_file(sentence1, sentence1_file_name)
write_on_file(sentence2, sentence2_file_name)

import csv

csv_file_name = 'data/sts_benchmark/sts-train.csv'
tr1_file_name = 'data/sts_benchmark/tr1'
tr2_file_name = 'data/sts_benchmark/tr2'
scores_file_name = 'data/sts_benchmark/scores'

with open(csv_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        if line_count == 2000:
            break
        with open(scores_file_name, 'a') as file:
            file.write(row[4] + '\n')
        with open(tr1_file_name, 'a') as file:
            file.write(row[5] + '\n')
        with open(tr2_file_name, 'a') as file:
            file.write(row[6] + '\n')
        line_count += 1




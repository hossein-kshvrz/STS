prefix = 'T2A'
suffix = '.txt'

filenames = [prefix + str(i) + suffix for i in range(0, 14851)]

for file in filenames:
    with open('T2A_newspaper/new/'+file, 'w') as outfile:
        with open('T2A_newspaper/'+file, encoding='cp1256') as infile:
            for line in infile:
                outfile.write(line)

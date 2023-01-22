from packet import Packet
import csv

arr = []
with open("data/UNSW-NB15-BALANCED-TRAIN.csv", mode='r') as file:
    # reading the CSV file
    csvFile = csv.reader(file)
    # displaying the contents of the CSV file
    for line in csvFile:
        if line[0] == "srcip":
            continue
        arr.append(Packet(line[0], line[1], line[2], line[3], line[4], line[5],
                          line[6], line[7], line[8], line[9], line[10], line[11],
                          line[12], line[13], line[14], line[15], line[16], line[17],
                          line[18], line[19], line[20], line[21], line[22], line[23],
                          line[24], line[25], line[26], line[27], line[28], line[29],
                          line[30], line[31], line[32], line[33], line[34], line[35],
                          line[36], line[37], line[38], line[39], line[40], line[41],
                          line[42], line[43], line[44], line[45], line[46], line[47],
                          line[48]))

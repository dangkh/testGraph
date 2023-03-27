import os
import numpy as np
import argparse
import pandas as pd
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extraction Result')
    parser.add_argument('--inputDir', help='input directory')
    parser.add_argument('--outputFile', help='file name contain extracted result', default='./result.txt')
    parser.add_argument('--csvFile', help='csv file name contain extracted result', default='./result.csv')
    parser.add_argument('--exportCsv', action='store_true',default=False, help='file csv contain extracted result')
    listRes = []
    listTypes = []
    maxAcc, varAcc = [], []
    args = parser.parse_args()
    print(args)
    f = open(args.inputDir, "r")
    for line in f:
        if line[:7] == 'Highest':
            infos = line.split(',')
            hAcc = float(infos[0].split(':')[-1])
            if types not in listTypes:
                listTypes.append(types)
                listRes.append([])
            indexType = listTypes.index(types)
            listRes[indexType].append(hAcc)
        elif line[2:10] == 'numEpoch':
            infos = line.split(',')
            types =  int(infos[3].split(':')[-1])
            # print(infos[29], infos[30])
            # print(types)
# print(listRes)
for tt in range(len(listRes)):
    # print(listTypes[tt], listRes[tt])
    maxAcc.append(np.mean(listRes[tt]))
    varAcc.append(np.std(listRes[tt]))
print(maxAcc)
print(varAcc)
fileNameET = args.csvFile
if args.exportCsv != 'False':
    with open(fileNameET, mode='w', newline='', encoding='utf-8') as ETfile:
        orders = ['0','10', '20', '30', '40', '50']
        acc = [maxAcc[listTypes.index(int(index))] for index in orders]
        var = [varAcc[listTypes.index(int(index))] for index in orders]
        fieldnames = ['Type'] + orders
        et_writer = csv.writer(ETfile)
        et_writer.writerow(fieldnames)
        et_writer.writerow(['Highest Acc: '] + acc)
        et_writer.writerow(['var: '] + var)
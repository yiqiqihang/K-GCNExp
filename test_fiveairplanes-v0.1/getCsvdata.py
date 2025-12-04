import csv

import numpy as np


def strtolist(str_one):
    if str_one =='[]':
        return str_one
    Long1 = str_one
    Long1 = Long1[1:-1]
    Long1 = Long1.replace("'", "")
    Long1 = Long1.split(",")
    b1 = np.array(Long1, dtype=np.float64)
    return b1
def strtolistint(str_one):
    Long1 = str_one
    Long1 = Long1[1:-1]
    Long1 = Long1.replace("'", "")
    Long1 = Long1.split(",")
    b1 = np.array(Long1, dtype='int')
    return b1

def get_csv_data(allData,label):

    path = r"E:\PostGraduate\su_master\dataset\20230918.csv"
    with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    label.append(int(row[2][1:len(row[2])]))
                    allData["filename"].append(row[1])
                    allData["Type"].append(int(row[2][1:len(row[2])]))
                    # allData["Frame1"].append(row[3])
                    # allData["Frame2"].append(row[4])
                    allData["Longitude1"].append(strtolist(row[5]).tolist())
                    allData["Longitude2"].append(strtolist(row[8]).tolist())
                    allData["Latitude1"].append(strtolist(row[6]).tolist())
                    allData["Latitude2"].append(strtolist(row[9]).tolist())
                    allData["Altitude1"].append(strtolist(row[7]).tolist())
                    allData["Altitude2"].append(strtolist(row[10]).tolist())

                    allData['ReadltarRng0'].append(strtolist(row[17]).tolist())
                    allData['ReadltarRng1'].append(strtolist(row[18]).tolist())
                    allData['ReadltarRng2'].append(strtolist(row[19]).tolist())
                    allData['ReadltarRng3'].append(strtolist(row[20]).tolist())


    return allData,label

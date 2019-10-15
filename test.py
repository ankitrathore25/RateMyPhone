import pandas as pd
import numpy as np
from numpy.linalg import norm

import fileinput, sys, csv

# data = pd.read_csv("testFile.csv") 

# data.head()

if __name__=="__main__":
    arr = [1,23,6,767,13,76,234,8967,234,8,34,8,34]
    #this will return the index of top 10 elements of the array, arranged in decreasing order
    sorted_arr = np.argsort(arr)[::-1][:10] 
    indexList = []
    print(sorted_arr)
    print(list(sorted_arr))
    for index in sorted_arr:
        print(index)
        # indexList.append(index)
    # print(arr)
    # print(indexList)
    i = '30'
    dictionary = {"1":"one","2":"two","3":"three","4":"four"}
    somedic = {"1":1,"2":2,"3":3}
    array = set()
    array.add(dictionary.get("1"))
    array.add(dictionary.get("2"))
    array.add(dictionary.get("2"))
    print(array)

    # # reader = csv.reader(open('amazonPhoneData.csv', 'rb'))
    # # reader1 = csv.reader(open('output1.csv', 'rb'))
    # writer = csv.writer(open('output1.csv', 'wb'))
    with open('amazonPhoneDataset.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        writer = csv.writer(open('output1.csv', 'w'))
        #[0:'Id', 1:'Product_name', 2:'by_info', 3:'Product_url', 4:'Product_img', 5:'Product_price', 6:'rating', 
        # 7;'total_review', 8:'ans_ask', 9:'prod_des', 10:'feature', 11:'cust_review']
        for row in readCSV:
            fileData = row[1] + ' ' + row[9] + ' ' + row[10] #this will append multiple text column into one
            if len(fileData.strip()) > 0:
                writer.writerow(row)

    # reader = csv.reader(open('amazonPhoneDataset.csv', 'rb', encoding='utf-8'))
    # # reader1 = csv.reader(open('output1.csv', 'rb'))
    # writer = csv.writer(open('output1.csv', 'wb'))
    # for row in reader:
    #     # row1 = reader1.next()
    #     if len(row[1]+row[9]+row[10]) > 0:
    #         writer.writerow(row)


    # with open('amazonPhoneData', 'rb') as file_b:
    #     r = csv.reader(file_b)
    #     next(r) #skip header
    #     rowData = {row[1]+ row[9]+row[10] for row in r}
    #     if len(rowData) > 0:
    #         w.writerow(row)
    #     f = fileinput.input('output1', inplace=True) # sys.stdout is redirected to the file
    #     # print(next(f)) # write header as first line

    #     w = csv.writer(sys.stdout) 
    #     for row in csv.reader(f):
    #     if (row[0], row[2]) in seen: # write it if it's in B
    #         w.writerow(row)
    # print(somedic.keys())
    # print(somedic.values())
    # print(norm([3,4]))
    # print(dictionary.items())
    # if i in dictionary:
    #     print(i)
    # else:
    #     print("Not in array")

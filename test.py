import pandas as pd
import numpy as np
from numpy.linalg import norm

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
    print(somedic.keys())
    print(somedic.values())
    print(norm([3,4]))
    print(dictionary.items())
    if i in dictionary:
        print(i)
    else:
        print("Not in array")

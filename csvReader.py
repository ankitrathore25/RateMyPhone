import csv

def readCSVFile() :
    dataset = {}
    # with open('amazon_phone_dataset.csv') as csvfile:
    with open('testFile.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        #[0:'Id', 1:'Product_name', 2:'by_info', 3:'Product_url', 4:'Product_img', 5:'Product_price', 6:'rating', 
        # 7;'total_review', 8:'ans_ask', 9:'prod_des', 10:'feature', 11:'cust_review']
        for row in readCSV:
            fileData = row[1] + ' ' + row[9] + ' ' + row[10] #this will append multiple text column into one
            if len(fileData.strip()) > 0:
                dataset.update({row[0]: fileData})
    global data 
    data = dataset

def testReadCSVFile() :
    dataset = {}
    with open('testFile2.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            fileData = row[1]
            if len(fileData.strip()) > 0:
                dataset.update({row[0]: fileData})
    print(len(dataset))
    global data 
    data = dataset
import pandas as pd
import json 

def processMainRecords(json_path,customer_id):
    file = open(json_path, 'r')
    data = json.load(file)
    file.close()
    if type(data) != list or len(data) == 0:
        print("No data to process!")
        return pd.DataFrame()
    else: 
        dict_data = {}
        for elem in data[0].keys():
            if type(data[0][elem]) == dict: 
                for subkey in data[0][elem].keys():
                    dict_data[elem + "_" + subkey] = []
            elif type(data[0][elem]) != list: 
                dict_data[elem] = []
        for elem in data:
            if ("customer_id" in dict_data.keys() and elem["customer_id"] == customer_id) or ("_id" in dict_data.keys() and elem["_id"] == customer_id):
                for key in elem.keys():
                    if type(elem[key]) == dict:
                        for subkey in elem[key].keys():
                            dict_data[key + "_" + subkey].append(elem[key][subkey])
                    elif type(elem[key]) != list: 
                        dict_data[key].append(elem[key])

        return pd.DataFrame(dict_data)
    
print(processMainRecords('../data/transactions.json',"customer_001").head())
print(processMainRecords("../data/investments.json","customer_001").head())
print(processMainRecords("../data/customers.json","customer_001").head())

def selection_json(values,iterate):
    for i in range(iterate):
        for value in values[i]["restaurants"]:
            item = {
                "event": value["restaurant"]["zomato_events"][0]["event"]["is_active"],
                }
        data.append(item)


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
data=[]
jsonFile = open('file1.json')
values = json.load(jsonFile)
jsonFile.close()
selection_json(values,75)
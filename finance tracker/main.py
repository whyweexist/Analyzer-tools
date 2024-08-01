import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_entry import get_date, get_amount, get_category, get_description
import warnings
warnings.filterwarnings('ignore')

class CSV:
    
    CSV_FILE = 'finance_data.csv'
    
    @classmethod
    def initialize(self):
        try:
            if not os.path.exists(self.CSV_FILE):
                self.create_csv()
        except Exception as e:
            print(f"Error in opening file {e}")

    @classmethod
    def create_csv(self):
        try:
            if not os.path.exists(self.CSV_FILE):
                print(f"Creating {self.CSV_FILE}")
                pd.DataFrame(columns=['date','amount','category','description']).to_csv(self.CSV_FILE, index=False)
            elif os.path.exists(self.CSV_FILE):
                print(f"File {self.CSV_FILE} already exists")
        except Exception as e:
            print(f"Error in creating file {e}")
    
    @classmethod    
    def read_csv(self):
        df = pd.read_csv(self.CSV_FILE)
        return df
    
    @classmethod
    def insert_data(self,date,amount,category,description):
        df = pd.read_csv(self.CSV_FILE)
        df = df._append({
                        'date':date,
                        'amount':amount,
                        'category':category,
                        'description':description
            },ignore_index=True)
        df.to_csv(self.CSV_FILE,index=False)
    
    @classmethod
    def create_pie_chart(self):
        df = pd.read_csv(self.CSV_FILE)
        df = df.groupby('category')['amount'].sum()
        df.plot(kind='pie',autopct='%1.1f%%')
        plt.show()
    
    @classmethod
    def line_chart(self):
        df = pd.read_csv(self.CSV_FILE)
        df = df.groupby(['date','category'])['amount'].sum()
        df.plot(kind='line')
        plt.show()



def add_data():
    CSV.initialize()
    date = get_date(
                    "Enter the date (DD-MM-YYYY) or enter for today's date: ", 
                    allow_default=True
                    )
    amount = get_amount()
    category = get_category()
    description = get_description()
    CSV.insert_data(date,amount,category,description)


add_data()

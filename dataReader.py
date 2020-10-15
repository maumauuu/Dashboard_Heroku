import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
from pandas import DataFrame


cd = pd.read_csv("data/carData.csv", )


def data_vis():
    # Chargement des données qui seront utilisées.
    cd.info()

    #print(cd.describe())

    #print(cd.head(10))

def create_table():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    #"cur.execute('CREATE TABLE CARS (Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)')
    #conn.commit()
    cd.to_sql('CARS', conn, if_exists='append', index=False)


def requete_price():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
    SELECT Car_Name, Selling_Price FROM CARS ORDER BY Selling_Price
              ''')

    df = DataFrame(cur.fetchall(), columns=['Car_Name', 'Price'])
    return df

    #sns.catplot(x='Car_Name', y='Price',
     #           data=df, jitter='0.25')



#cur.execute('''
#SELECT * FROM CARS
 #         ''')

#cur.execute('''
#SELECT Car_Name, max(Selling_Price) FROM CARS
#          ''')



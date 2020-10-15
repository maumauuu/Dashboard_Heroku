import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sqlite3
from pandas import DataFrame
import statsmodels.api as sm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from my_linear_regression import Linear_Regression
from sklearn.linear_model import LinearRegression
import plotly.express as px

cd = pd.read_csv("data/carData.csv", )


def data_vis():
    # Chargement des données qui seront utilisées.
    cd.info()
    # print(cd.describe())
    # print(cd.head(10))


def create_table():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    # "cur.execute('CREATE TABLE CARS (Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type,
    # Seller_Type, Transmission, Owner)') conn.commit()
    cd.to_sql('CARS', conn, if_exists='append', index=False)


def requete_price():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
    SELECT Car_Name, Selling_Price FROM CARS ORDER BY Selling_Price
              ''')

    df = DataFrame(cur.fetchall(), columns=['Car_Name', 'Price'])
    return df

    # sns.catplot(x='Car_Name', y='Price',
    #           data=df, jitter='0.25')


def reg_np():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
    SELECT Selling_Price,Year FROM CARS 
              ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    X = df_regression.Selling_Price
    y = df_regression.Year

    model = np.polyfit(X, y, 1)
    predict = np.poly1d(model)
    x_lin_reg = X
    y_lin_reg = predict(x_lin_reg)

    fig = px.scatter(X, y)
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title='selling price', type='linear')

    fig.update_yaxes(title='year', type='linear')
    fig.show()
    # plt.plot(x_lin_reg, y_lin_reg, c='r')

    return fig


def reg_sp():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    X = df_regression.Selling_Price
    y = df_regression.Year

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(X, y)
    plt.plot(X, y, 'o', label='original data')
    plt.plot(X, intercept + slope * X, 'r', label='fitted line')
    plt.legend()
    plt.show()


def reg_sk():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    # print(df_regression.shape)
    X = df_regression.iloc[:, :-1].values
    # print(X.shape)
    y = df_regression.iloc[:, 1].values
    # print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_test)
    plt.scatter(X, y)
    plt.plot(X_test, pred, c='r')

    df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
    # print(df)
    plt.show()


def reg_sk_multiple():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()

    cur.execute('''  
    SELECT Selling_Price,Kms_Driven,Transmission, Year FROM CARS 
              ''')

    dff = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Kms_Driven', 'Transmission', 'Year'])

    # Récupérer l'ensemble des valeurs de la variable cible
    Y = dff["Year"]
    # Récupérer les variables prédictives (on en a 2)
    X = dff[['Kms_Driven', 'Selling_Price']]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(dff["Kms_Driven"], dff["Selling_Price"], dff["Year"], c='r', marker='^')

    ax.set_zlabel('Year')
    ax.set_xlabel('Kms_Driven')
    ax.set_ylabel('Selling_Price')

    plt.show()

    scale = StandardScaler()
    # X_scaled = scale.fit_transform(X[['Kms_Driven', 'Selling_Price']].as_matrix())

    est = sm.OLS(Y, X).fit()


# print(est.summary())


def my_reg():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    X = df_regression.Selling_Price
    y = df_regression.Year

    lr = Linear_Regression()
    lr.fit(X, y)
    pred = lr.predict(X)

    plt.scatter(X, y)
    x1, x2, y1, y2 = plt.axis()
    plt.plot(X, pred, c='r')

    plt.axis((x1, x2, 2002, 2019))
    plt.show()


def svm_():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    X_ = df_regression.iloc[:, :-1].values
    # print(X.shape)
    y_ = df_regression.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.20)
    print('pass 2')

    clf = svm.SVC(kernel='linear')
    print('pass 3')

    clf.fit(X_train, y_train)
    print('pass 4')

    y_pred = clf.predict(X_test)

    plt.clf()

    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_pred)

    x1, x2, y1, y2 = plt.axis()

    plt.axis((x1, x2, 2002, 2019))
    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
    #            facecolors="none", zorder=10, edgecolors="k")
    plt.show()

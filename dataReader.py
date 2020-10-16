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
import seaborn as sns
from my_linear_regression import Linear_Regression
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objs as go


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
    SELECT Year, Selling_Price FROM CARS ORDER BY Selling_Price
              ''')

    df = DataFrame(cur.fetchall(), columns=['Year', 'Price'])

    sp = sns.catplot(x='Year', y='Price',
                     data=df, jitter='0.25')
    sp.savefig("assets/sea-plot.png")
    plt.show()
    #return df



def reg_np():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
    SELECT Selling_Price,Year FROM CARS 
              ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    y = df_regression.Selling_Price
    X = df_regression.Year

    model = np.polyfit(X, y, 1)
    predict = np.poly1d(model)
    x_lin_reg = X.iloc[:20]
    y_lin_reg = predict(x_lin_reg)

    trace1 = go.Scatter(x=X, y=y,
                        mode='markers')

    trace3 = go.Scatter(x=x_lin_reg, y=y_lin_reg,
                        mode='markers')

    layout = go.Layout(title='Quantification de l\'âge en fonction du prix de vente',
                       hovermode='closest',

                       )
    trace2= go.Scatter(x=x_lin_reg, y=y_lin_reg,
                       line=dict(width=2,
                                 color='rgb(255, 0, 0)'))

    return go.Figure(data=[trace1,trace2,trace3], layout=layout), 'Application de la régression linéaire en utilisant numpy\n' \
                                                                  'le tracé de la droite de régression n\'est pas trés significatif dans notre cas ' \
                                                                  'mais on voit bien que les prédictions effectuées sur un échantillon sont correctes.'




def reg_sp():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    y = df_regression.Selling_Price
    X = df_regression.Year

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(X, y)
    #plt.plot(X, y, 'o', label='original data')
    #plt.plot(X, intercept + slope * X, 'r', label='fitted line')
    #plt.legend()
    #plt.show()
    y_pred = intercept + slope * X

    trace1 = go.Scatter(x=X, y=y,
                        mode='markers')

    #trace3 = go.Scatter(x=x_lin_reg, y=y_lin_reg,
                  #      mode='markers')

    layout = go.Layout(title='Quantification de l\'âge en fonction du prix de vente',
                       hovermode='closest',

                       )
    trace2 = go.Scatter(x=X, y=y_pred,
                        line=dict(width=2,
                                  color='rgb(255, 0, 0)'))

    return go.Figure(data=[trace1,trace2],
                     layout=layout), 'Régression linéaire avec scipy.'


def reg_sk():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Year','Selling_Price', ])

    y = df_regression.Year.values
    X = df_regression.iloc[:, 1:].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_test)
    plt.scatter(X, y)
    plt.plot(X_test, pred, c='r')

    df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
    # print(df)
    #plt.show()
    trace1 = go.Scatter(x=X_train.flatten(), y=y_train,
                        mode='markers')

    trace3 = go.Scatter(x=X_test.flatten(), y=pred,
                        mode='markers')

    layout = go.Layout(title='Quantification de l\'âge en fonction du prix de vente',
                       hovermode='closest'
                       )

    trace2 = go.Scatter(x=X_test.flatten(), y=pred,
                        line=dict(width=2,
                                  color='rgb(255, 0, 0)'))

    return go.Figure(data=[trace1,trace2],
                     layout=layout), 'Régression linéaire avec Sickit-learn.'




def reg_sk_multiple():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()

    cur.execute('''  
    SELECT Selling_Price,Kms_Driven,Transmission, Year FROM CARS 
              ''')

    dff = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Kms_Driven', 'Transmission', 'Year'])

    # Récupérer l'ensemble des valeurs de la variable cible
    X = dff[["Year",'Kms_Driven']]
    # Récupérer les variables prédictives (on en a 2)
    y_ = dff['Selling_Price']

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    #ax.scatter(dff["Kms_Driven"], dff["Selling_Price"], dff["Year"], c='r', marker='^')

    #ax.set_zlabel('Year')
    #ax.set_xlabel('Kms_Driven')
    #ax.set_ylabel('Selling_Price')

    #plt.show()

    scale = StandardScaler()
    # X_scaled = scale.fit_transform(X[['Kms_Driven', 'Selling_Price']].as_matrix())

    est = sm.OLS(y_,X).fit()

    pred = est.predict(X)


    trace1 = go.Scatter3d(x=dff.Year, z=y_, y=dff.Kms_Driven,
                        mode='markers')

    trace3 = go.Scatter3d(x=dff.Year, z=pred, y=dff.Kms_Driven,
                        mode='markers')


    layout = go.Layout(title='Quantification de l\'âge en fonction du prix de vente',
                       hovermode='closest'
                       )


    #trace2 = go.Scatter(x=X_test.flatten(), y=pred,
        #                line=dict(width=2,
        #                 color='rgb(255, 0, 0)'))

    return go.Figure(data=[trace1, trace3],
                     layout=layout), 'Régression linéaire multiple avec Sickit-learn.'


# print(est.summary())


def my_reg():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
        SELECT Selling_Price,Year FROM CARS 
                  ''')

    df_regression = DataFrame(cur.fetchall(), columns=['Selling_Price', 'Year'])

    X = df_regression.Year.values
    y = df_regression.Selling_Price.values


    lr = Linear_Regression()
    lr.fit(X, y)
    X_test = X[:20]
    pred = lr.predict(X_test)

    #plt.scatter(X, y)
    #x1, x2, y1, y2 = plt.axis()
    #plt.plot(X, pred, c='r')

    #plt.axis((x1, x2, 2002, 2019))
    #plt.show()

    trace1 = go.Scatter(x=X, y=y,
                          mode='markers')

    trace3 = go.Scatter(x=X_test, y=pred,
                          mode='markers')

    layout = go.Layout(title='Quantification de l\'âge en fonction du prix de vente',
                       hovermode='closest'
                       )

    # trace2 = go.Scatter(x=X_test.flatten(), y=pred,
    #                line=dict(width=2,
    #                          color='rgb(255, 0, 0)'))

    return go.Figure(data=[trace1,trace3],
                     layout=layout), 'Régression linéaire multiple avec Sickit-learn.'


def svm_():
    conn = sqlite3.connect('data/car.db')
    cur = conn.cursor()
    cur.execute('''  
            SELECT Selling_Price,Year FROM CARS 
                      ''')
    df_regression = DataFrame(cur.fetchall(), columns=['Year', 'Selling_Price'])

    y_ = df_regression.Year.values
    X_ = df_regression.iloc[:, 1:].values
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.20)

    clf = svm.SVR(kernel='linear')

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # plt.clf()

    #plt.scatter(X_train, y_train)
    #plt.scatter(X_test, y_pred)

    #x1, x2, y1, y2 = plt.axis()

    #   plt.axis((x1, x2, 2002, 2019))
    # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
    #            facecolors="none", zorder=10, edgecolors="k")
    #plt.show()

    trace1 = go.Scatter(x=X_train.flatten(),y=y_train,
                          mode='markers')

    trace3 = go.Scatter(x=X_test.flatten(), y=y_pred,
                          mode='markers')

    layout = go.Layout(title='Quantification de l\'âge en fonction du prix de vente',
                       hovermode='closest'
                       )

    trace2 = go.Scatter(x=X_test.flatten(), y=y_pred,
                    line=dict(width=2,
                              color='rgb(255, 0, 0)'))

    return go.Figure(data=[trace1, trace2, trace3],
                     layout=layout), 'Régression linéaire multiple avec Sickit-learn.' \
                                     '\n Contrairement à la régression linéaire qui minimise l\'erreur totale, ' \
                                     'SVM essaie de miniser les marges en trouvant les meilleures vecteurs support.'


if __name__ == '__main__':
    requete_price()

### Heroku
Le projet est hébergé sur Heroku :

 https://visualistaion-tp.herokuapp.com/ 
 
Pour l'éxécuter en local, il faut lancer le fichier index.py
    
## Projet

Présentation d'un dashboard exploitant
les données de carData.csv.

Choix de la voiture à acheté en étudiant les différents prix du marché.

![Screenshot](img/shot.gif)


### Base de données

J'ai utilisé Sqlite3 afin de créer une base de données local et faire des requêtes 
dessus.

Toutes les données pour effectuer les plots, les regressions linéaires... sont 
récupérées via des requêtes sql.

### Regression linéaire

La régression linéaire est une méthode permettant de découvrir la 
relation entre deux variables de l'ensemble de données,
 telles que le prix de la voiture et l'année de fabrication. (cf sujet)
 
 
J'effectue une étude de la quantification de l'année de vente de la voiture
par rapport au prix de vente en utilisant les algorithmes 
de régression linéaire (numpy, scipy, sklearn)
et en créant le notre.

Comparaison entre tous ces algorithmes de regréssion et avec SVM.

Tous les algorithmes(regression et svm) donnent des résultats similaires, on nottera un temps beaucoup
plus long pour SVM et la classe de regression.

### Important
L'algorithme de SVM dépasse le timeout sue Heroku, il ne s'affichera donc pas. Il marche en local.

## Architecture logiciel
Le projet à été effectué sur ubuntu 18.04 avec
 - [Python][1] (3.8) 
 - [Dash][2] (1.16.2)
 - [scikit-learn][3] (0.23.2)

[1]: https://www.python.org/download/releases/3./
[2]: https://dash.plotly.com/installation
[3]: https://scikit-learn.org/stable/install.html
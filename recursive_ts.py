# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 09:03:44 2023

@author: kieu_
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


data = pd.read_csv(r"C:\Users\kieu_\OneDrive\Desktop\Project\Data Science\VietNguyen\ML_9-1\Datasets\co2.csv")
data["time"] = pd.to_datetime(data["time"])


# Compléter les données manquantes (missing values)
data["co2"] = data["co2"].interpolate() # interpolate (au lieu de mean, median ou mode) est utilisé dans les time series


# Afficher une représentation graphique de la série temporelle
fig, ax = plt.subplots()
ax.plot(data["time"], data["co2"])
ax.set_xlabel("time")
ax.set_ylabel("CO2")
plt.show()



### Sur la base de CO2 sur 5 semaines, on va prédire le CO2 de la 6ème semaine
"""
Pour cela, on va d'abord créer 5 nouvelles colonnes basées sur la colonne CO2. C'est le principe du recursive multi-step time series forecasting.
On va donc avoir la colonne CO2 d'origine, puis 4 nouvelles colonnes CO2 en décalant les valeurs de CO2, puis la dernière colonne est target.
"""


# Recursive multi-step Time series forcasting

# Créer de nouvelles colonnes CO2 en décalant les valeurs CO2
def create_recursive_data(data, window_size, target_name):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data[target_name] = data["co2"].shift(-i)
    
    # Enlever les na situés à la fin du dataset, cela est lié au décalage vers les nouvelles colonnes CO2
    data = data.dropna(axis=0)
    
    return data


target = "target"
window_size = 5 # on utilise le CO2 de 5 semaines pour prédire la 6ème semaine
data = create_recursive_data(data, window_size, target)


X = data.drop([target, "time"], axis=1) # on enlève la colonne "target" mais aussi celle "time"
y = data[target]


# Dans les time series, la division en train et test se fait différemment en respectant l'ordre chronologique (80% des données du début pour le train, puis 20% dernières données en test)
# Il ne faut pas utiliser train_test_split qui divise les données de manière aléatoire
train_size = 0.8
num_samples = len(X)

X_train = X[:int(num_samples * train_size)]
y_train = y[:int(num_samples * train_size)]
X_test = X[int(num_samples * train_size):]
y_test = y[int(num_samples * train_size):]



### On va tester le modèle Linear Regression, car la corrélation est très grande entre les features
corr_mat = data.corr()
print(corr_mat)


reg = LinearRegression()
reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)
print("R2: {}".format(r2_score(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))

# R2: 0.9907505918201436 # au max et s'approche de 1
# MSE: 0.22044947360346612 # très faible
# MAE: 0.3605603788359238
# Le modèle est très bon !


for i, j in zip(y_predict, y_test):
    print("Prediction: {}. Actual value: {}".format(i, j))


fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples * train_size)], data["co2"][:int(num_samples * train_size)], label="Train")
ax.plot(data["time"][int(num_samples * train_size):], data["co2"][int(num_samples * train_size):], label="Test")
ax.plot(data["time"][int(num_samples * train_size):], y_predict, label="Prediction")
ax.set_xlabel("time")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()


# Le modèle RandomForestRegressor donne de mauvais résultats
# Par principe il donne le résulat final en fonction de la moyenne des n arbres
# Ob observe que les prédictions ne dépassent pas la valeur max du train
# En conséquence, il n'est pas capable de donner des prédictions en dehors du range qu'il a vu dans le train (valeurs jamais rencontrées)



"""
POUR ALLER PLUS LOIN

A utiliser lazypredict pour trouver le modèle le plus adapté
A utiliser GridSearchCV pour trouver les meilleurs paramètres
"""



# Predict for new data
current_data = [380.5, 390, 390.2, 390.2, 391.3]

for i in range(10):
    print(current_data)
    prediction = reg.predict([current_data]).tolist()
    print("CO2 in week {} is {}".format(i+1, prediction[0]))
    current_data = current_data[1:] + prediction
    print("-------------------------")


# [380.5, 390, 390.2, 390.2, 391.3]
# CO2 in week 1 is 393.01537625192634
# -------------------------
# [390, 390.2, 390.2, 391.3, 393.01537625192634]
# CO2 in week 2 is 393.06413252834716
# -------------------------
# [390.2, 390.2, 391.3, 393.01537625192634, 393.06413252834716]
# CO2 in week 3 is 393.5501337857559
# -------------------------
# [390.2, 391.3, 393.01537625192634, 393.06413252834716, 393.5501337857559]
# CO2 in week 4 is 394.21445167014474
# -------------------------
# [391.3, 393.01537625192634, 393.06413252834716, 393.5501337857559, 394.21445167014474]
# CO2 in week 5 is 394.5220187068547
# -------------------------
# [393.01537625192634, 393.06413252834716, 393.5501337857559, 394.21445167014474, 394.5220187068547]
# CO2 in week 6 is 394.69201188532054
# -------------------------
# [393.06413252834716, 393.5501337857559, 394.21445167014474, 394.5220187068547, 394.69201188532054]
# CO2 in week 7 is 394.9601683413949
# -------------------------
# [393.5501337857559, 394.21445167014474, 394.5220187068547, 394.69201188532054, 394.9601683413949]
# CO2 in week 8 is 395.130823657103
# -------------------------
# [394.21445167014474, 394.5220187068547, 394.69201188532054, 394.9601683413949, 395.130823657103]
# CO2 in week 9 is 395.20846115977514
# -------------------------
# [394.5220187068547, 394.69201188532054, 394.9601683413949, 395.130823657103, 395.20846115977514]
# CO2 in week 10 is 395.2829261281513
# -------------------------


prediction = reg.predict([current_data])
print(prediction)
# [395.33439155]

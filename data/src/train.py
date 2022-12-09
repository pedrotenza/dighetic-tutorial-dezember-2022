import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv('auto-mpg-training-data.csv', sep=";")

print(data)

data = data.sample(frac=1)

y_variable = data[mpg]

x_variables = data.loc[:, data.columns != 'mpg']


X_train, X_test, Y_train, Y_test = (x_variables, y_variable)

regressor = LogisticRegression()
regressor .fit(X_train, Y_train)


y_pred = regressor.predrict(X_test)

# file_to_write = open("data/models/baummethoden_lr.pickle, "wb")

#pickle.dump(regressor, file_to_write)

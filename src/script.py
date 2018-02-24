import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_linreg(intercept, normalise):
    return LinearRegression(fit_intercept=intercept, normalize=normalise, n_jobs=-1)


def linreg_predict(X, y, test_input, intercept=False, normalise=True):
    print("Get linear regression prediction")

    linreg = get_linreg(intercept, normalise)
    linreg.fit(X, y)

    return linreg.predict(test_input)


def polyreg_predict(X, y, test_input, degrees, intercept=False, normalise=True):
    print("Get polynomial regression prediction")

    linreg = get_linreg(intercept, normalise)
    polyreg = PolynomialFeatures(degree=degrees)

    x_hat = polyreg.fit_transform(X)
    predict_hat = polyreg.fit_transform(test_input)

    linreg.fit(x_hat, y)

    return linreg.predict(predict_hat)


EnergySet = pd.read_csv("data/energydata_complete.csv")
df = pd.DataFrame(data=EnergySet)

output = df["Appliances"]
input = df.loc[0:, "lights":"RH_9"]
test_input = df.loc[:1000, "lights":"RH_9"]
poly_degrees = 3




correlation_matrix = df.corr()

#print(correlation_matrix["Appliances"].sort_values(ascending=False))

print(df["T_out"].describe())



'''
for i in range(2, len(df.columns)):
    ax = df.plot.scatter(df.columns[i], df.columns[1], alpha=0.1)
    fig = ax.get_figure()
    fig.savefig("plots/" + str(df.columns[i]) + "-plot.png")
'''


'''
print(input.shape)

lin_predictions = linreg_predict(input, output, test_input)
poly_predictions = polyreg_predict(input, output, test_input, poly_degrees)

print("Linear error = " + str(math.sqrt(mean_squared_error(df.loc[:1000, "Appliances"], lin_predictions))))
print("Polynomial error = " + str(math.sqrt(mean_squared_error(df.loc[:1000, "Appliances"], poly_predictions))))
'''


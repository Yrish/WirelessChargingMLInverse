import numpy as np
from sklearn.linear_model import LinearRegression
from utils import Data
from cost_evaluation import getTestData
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

dat = Data(switchInputAndOutput = True)
X, Y = dat.getAllItems()
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

reg = make_pipeline(PolynomialFeatures(6), LinearRegression())
reg = reg.fit(X, Y)
print(reg.score(X, Y))


getTestData(reg, 10, isNeuralNetwork=False)

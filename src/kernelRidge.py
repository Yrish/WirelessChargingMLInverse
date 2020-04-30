import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from utils import Data
from cost_evaluation import getTestData

kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
dat = Data(switchInputAndOutput = True)
X, Y = dat.getAllItems()
X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

startTime = time.time()
kr.fit(X, Y)
elapse_time = time.time() - startTime
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % elapse_time)

print(kr.score(X, Y))
getTestData(kr, 10, isNeuralNetwork=False)

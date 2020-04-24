import random
from cost import NeuralNet
from torch import from_numpy, load
import numpy as np



headers = ["Iin[A]","Iout[A]","l[mm]","p1[mm]","p2[mm]","p3[mm]","win[mm]","kdiff[%]","Bleak[uT]","V_PriWind[cm3]","V_PriCore[cm3]","Pout[W]"]

# 0,   10,10,400,10,10,10,100,6.551478722,2.9,38.16,800,50.31296542
# 6326,40,40,700,60,60,60,700,0.526595281,61.11868038,140.76,2450,720.1673379

def random_in_range(minimum, maximum):
    return random.random() * (maximum - minimum) + minimum

generators = [
    lambda: random_in_range(10, 40),
    lambda: random_in_range(10, 40),
    lambda: random_in_range(400, 700),
    lambda: random_in_range(10, 60),
    lambda: random_in_range(10, 60),
    lambda: random_in_range(10, 60),
    lambda: random_in_range(100, 700),
    lambda: random_in_range(0, 10),
    lambda: random_in_range(0, 15),
    lambda: random_in_range(1, 150),
    lambda: random_in_range(1, 6000),
    lambda: random_in_range(1, 10000),
]

def getTestData(model, count):
    testInput = []
    output = []
    for i in range(count):
        inp = [generators[x]() for x in range(7, 12)]
        out = model(from_numpy(np.array(inp, dtype='f')))
        out = [abs(item.item()) for item in list(out.detach())]
        testInput.append(inp)
        output.append(out)
    with open("simulationOut.csv", "w") as simulationOut:
        simulationOut.write(",".join(headers[7:12]) + "\n")
        for i in range(len(testInput)):
            simulationOut.write(",".join([str(s) for s in testInput[i]]) + "\n")
    with open("predictedSimulationIn.csv", "w") as simulationIn:
        simulationIn.write(",".join(headers[:7]) + "\n")
        for i in range(len(output)):
            simulationIn.write(",".join([str(s) for s in output[i]]) + "\n")
    with open("allData.csv", "w") as allData:
        allData.write(",".join(headers) + "\n")
        for i in range(len(output)):
            allData.write(",".join([str(s) for s in output[i]]))
            allData.write(",".join([str(s) for s in testInput[i]]) + "\n")

if __name__ == '__main__':
    modelName = "../model.model"
    model = NeuralNet()
    model.load_state_dict(load(modelName))
    getTestData(model, 100)

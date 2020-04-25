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
    lambda: random_in_range(1, 100),
    lambda: random_in_range(1, 100),
    lambda: random_in_range(1, 1000),
    lambda: random_in_range(3, 150),
    lambda: random_in_range(3, 150),
    lambda: random_in_range(3, 150),
    lambda: random_in_range(3, 1500),
    lambda: random_in_range(1, 300),
    lambda: random_in_range(1, 10000),
    lambda: random_in_range(0.1, 40),
    lambda: random_in_range(1, 3000),
    lambda: random_in_range(1, 300),
]

def isPositive(num):
    return num > 0

impossible = [
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
    isPositive,
]

def betweenInclusive(minimum, maximum):
    return lambda x: x >= minimum and x <= maximum

restricted = [
    betweenInclusive(1, 100),
    betweenInclusive(1, 100),
    betweenInclusive(1, 1000),
    betweenInclusive(3, 150),
    betweenInclusive(3, 150),
    betweenInclusive(3, 150),
    betweenInclusive(3, 1500),
    betweenInclusive(1, 300),
    betweenInclusive(1, 10000),
    betweenInclusive(0.1, 40),
    betweenInclusive(1, 3000),
    betweenInclusive(1, 300),
]

print(len(headers))
print(len(restricted))
print(len(impossible))

def getTestData(model, count):
    restrictedCount = 0
    impossibleCount = 0
    testInput = []
    output = []
    while len(output) < count:
        inp = [generators[x]() for x in range(7, 12)]
        out = model(from_numpy(np.array(inp, dtype='f')))
        out = [item.item() for item in list(out.detach())]
        for i in range(len(out)):
            if impossible[i](out[i]):
                impossibleCount += 1
                continue
            if restricted[i](out[i]):
                restrictedCount += 1
                continue
        testInput.append(inp)
        output.append(out)
    print("Generated {} usable outputs, {} were restricted, {} were impossible".format(len(output), restrictedCount, impossibleCount))
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
    modelName = "../saves/biasCost/error_1449646.model"
    model = NeuralNet()
    model.load_state_dict(load(modelName))
    getTestData(model, 100)

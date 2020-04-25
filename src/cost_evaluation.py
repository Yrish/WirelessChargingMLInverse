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

def notFunction(function):
    return lambda x: not function(x)

impossible = [
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
    notFunction(isPositive),
]

def betweenInclusive(minimum, maximum):
    return lambda x: x >= minimum and x <= maximum

restricted = [
    notFunction(betweenInclusive(1, 100)),
    notFunction(betweenInclusive(1, 100)),
    notFunction(betweenInclusive(1, 1000)),
    notFunction(betweenInclusive(3, 150)),
    notFunction(betweenInclusive(3, 150)),
    notFunction(betweenInclusive(3, 150)),
    notFunction(betweenInclusive(3, 1500)),
    notFunction(betweenInclusive(1, 300)),
    notFunction(betweenInclusive(1, 10000)),
    notFunction(betweenInclusive(0.1, 40)),
    notFunction(betweenInclusive(1, 3000)),
    notFunction(betweenInclusive(1, 300)),
]

print(len(headers))
print(len(restricted))
print(len(impossible))

def getTestData(model, count, isNeuralNetwork=True):
    restrictedCount = 0
    impossibleCount = 0
    iterations = 0
    testInput = []
    output = []
    while len(output) < count:
        iterations += 1
        inp = [generators[x]() for x in range(7, 12)]
        out = None
        if isNeuralNetwork:
            out = model(from_numpy(np.array(inp, dtype='f')))
            out = [item.item() for item in list(out.detach())]
        else:
            out = model.predict(np.array([inp], dtype='f'))[0]
        breaking = False
        for i in range(len(out)):
            if impossible[i](out[i]):
                impossibleCount += 1
                breaking = True
                break
            if restricted[i](out[i]):
                restrictedCount += 1
                breaking = True
                break
        if iterations % 10000 == 0:
            print("done {} iterations, {} usable, {} impossible, {} restricted".format(iterations, len(output), impossibleCount, restrictedCount))
        if breaking:
            continue
        testInput.append(inp)
        output.append(out)
        print("generating {} {} {}".format(len(testInput), impossibleCount, restrictedCount))
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
    # modelName = "./saves/cost_e_3762065.8720703125_c_2766072.1286621094_ep_99.model"
    modelName = "./saves/cost_e_1533793.0595703125_c_1209800.4447021484_ep_6199.model"
    model = NeuralNet()
    model.load_state_dict(load(modelName))
    getTestData(model, 100)

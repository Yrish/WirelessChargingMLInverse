import pandas as pd
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np

class Data(Dataset):
    def __init__(self, csv_path: str = '../data/4th_dataset.csv', startingIndex: int = 1, inputCount: int = 7, switchInputAndOutput: bool = False) -> "Data":
        self.inputCount: int = inputCount
        self.startingIndex: int = startingIndex
        self.switchInputAndOutput: bool = switchInputAndOutput
        self.data = pd.read_csv(csv_path)
        self.outputCount = self.data.count(axis=1).iloc[0] - startingIndex - inputCount
        if switchInputAndOutput:
            tmp = self.outputCount
            self.outputCount = self.inputCount
            self.inputCount = tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        startingIndex = self.startingIndex
        inputEndIndex = self.startingIndex + self.inputCount
        input = from_numpy(np.array(self.data.iloc[idx, startingIndex:inputEndIndex])).float()
        output = from_numpy(np.array(self.data.iloc[idx, inputEndIndex:])).float()
        if self.switchInputAndOutput:
            return input, output
        return output, input

    def getRegularItem(self, idx):
        startingIndex = self.startingIndex
        inputEndIndex = self.startingIndex + self.inputCount
        input = self.data.iloc[idx, startingIndex:inputEndIndex]
        output = self.data.iloc[idx, inputEndIndex:]
        if self.switchInputAndOutput:
            return input, output
        return output, input

    def getAllItems(self):
        X = []
        Y = []
        for i in range(len(self)):
            x, y = self.getRegularItem(i)
            X.append(x)
            Y.append(y)
        return X, Y



if __name__ == '__main__':
    data = Data()
    print(data[0])

import pandas as pd
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np

# CSV: 7 nn inputs then 5 nn outputs
# 5 nn outputs: V_PriWind, kdiff, Bleak, Pout, Vcore
class Data(Dataset):
    def __init__(self, csv_path: str = '../data/data.csv', startingIndex: int = 1, inputCount: int = 7, switchInputAndOutput: bool = False) -> "Data":
        self.inputCount: int = inputCount
        self.startingIndex: int = startingIndex
        self.switchInputAndOutput: bool = switchInputAndOutput
        self.data = pd.read_csv(csv_path)
        self.outputCount = self.data.count(axis=1).iloc[0] - startingIndex - inputCount
        #print("Debug: self.outputCount is", self.outputCount)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        startingIndex = self.startingIndex
        inputEndIndex = self.startingIndex + self.inputCount
        input = from_numpy(np.array(self.data.iloc[idx, startingIndex:inputEndIndex])).float()
        output = from_numpy(np.array(self.data.iloc[idx, inputEndIndex:])).float()
        if self.switchInputAndOutput:
            return output, input
        return input, output


if __name__ == '__main__':
    data = Data()
    print(data[0])

import numpy as np
import re
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelBinarizer


class dataset(Dataset):
    catagoris = ['alouatta_palliata','erythrocebus_patas','cacajao_calvus','macaca_fuscata','cebuella_pygmea','cebus_capucinus','mico_argentatus','saimiri_sciureus','aotus_nigriceps','trachypithecus_johnii']
    def __init__(self, data_dir, transorm=None):
        super(dataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transorm
        lb = LabelBinarizer()
        lb.fit(self.catagoris)
        self.X, self.y = list(), list()
        for i in os.listdir(self.data_dir):
            self.X.append(i)
            str = re.search('^n\d',i)
            match str.group():
                case 'n0':
                    self.y.append(lb.transform(['alouatta_palliata']))
                case 'n1':
                    self.y.append(lb.transform(['erythrocebus_patas']))
                case 'n2':
                    self.y.append(lb.transform(['cacajao_calvus']))
                case 'n3':
                    self.y.append(lb.transform(['macaca_fuscata']))
                case 'n4':
                    self.y.append(lb.transform(['cebuella_pygmea']))
                case 'n5':
                    self.y.append(lb.transform(['cebus_capucinus']))
                case 'n6':
                    self.y.append(lb.transform(['mico_argentatus']))
                case 'n7':
                    self.y.append(lb.transform(['saimiri_sciureus']))
                case 'n8':
                    self.y.append(lb.transform(['aotus_nigriceps']))
                case 'n9':
                    self.y.append(lb.transform(['trachypithecus_johnii']))
    def __getitem__(self, item):
        if self.transform:
            X = self.transform(Image.open(os.path.join(self.data_dir,self.X[item])))
        else:
            X = Image.open(os.path.join(self.data_dir,self.X[item]))
        y = self.y[item]
        Y = y.reshape(-1)
        return X, Y.astype(np.float)
    def __len__(self):
        return len(self.y)

import pandas as pd
import torch
import collections


f = open("", 'rb')
data = torch.load(f, map_location='cpu')

for i in data.keys():
    k_ = i.split("module.")[1]
    data = collections.OrderedDict([(k_, v) if k == i else (k, v) for k, v in data.items()])
f1 = open("", 'wb')
torch.save(data, f1)

import os
import random

import numpy
import json
import collections



l = ["episodes_rewards", "evaluate_itr", "win_rates"]
p = ""
p1 = p + ""
os.makedirs(p1)
















for i in l:
    data = numpy.load(p + i + ".npy")
    d = {0: data.tolist()}
    with open(p1 + i + ".json", "w") as file:
        json.dump(d, file)

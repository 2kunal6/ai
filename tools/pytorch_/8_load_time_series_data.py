import numpy as np
import torch
import torch.nn.functional as F

bikes_numpy = np.loadtxt("resources/hour-fixed.csv", dtype=np.float32, delimiter=",", skiprows=1,
                         converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy)
print(bikes)

weather_one_hot_encoded = F.one_hot(bikes[:, 10].long(), 4)
print(weather_one_hot_encoded)

daily_bikes = torch.cat((bikes))
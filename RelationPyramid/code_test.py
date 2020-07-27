from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
torch.set_printoptions(edgeitems=10000)
steps=[8, 16, 32, 64, 100, 300]
image_size=300
feature_maps=[38,19,10,5,3,1]
min_dim= 300
steps= [8, 16, 32, 64, 100, 300]
min_sizes= [30, 60, 111, 162, 213, 264]
max_sizes= [60, 111, 162, 213, 264, 315]
aspect_ratios= [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
variance= [0.1, 0.2]
mean=[]
for k, f in enumerate(feature_maps):  # k=0,1,2,3,4,5, f=38,19,10,5,3,1
    for i, j in product(range(f), repeat=2):  # 멍미이거..?
        f_k = image_size / steps[k]  # get 38,19,10,5,3,1
        # unit center x,y
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k
        # aspect_ratio: 1
        # rel size: min_size
        s_k = min_sizes[k] / image_size
        mean += [cx, cy, s_k, s_k]

        # aspect_ratio: 1
        # rel size: sqrt(s_k * s_(k+1))
        s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))
        mean += [cx, cy, s_k_prime, s_k_prime]

        # rest of aspect ratios
        for ar in aspect_ratios[k]:
            mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
            mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

output = torch.Tensor(mean).view(-1, 4)
print(output.size())
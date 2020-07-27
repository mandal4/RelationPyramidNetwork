
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

test=torch.randn(2,8732,4)
test=test.tolist()
separated_priors=[]
#separated_priors=torch.Tensor(separated_priors)

anchor_size_per_source=[1*1*4, 3*3*4, 5*5*6, 10*10*6, 19*19*6, 38*38*4]
step=0
step=int(0)
for i in range(6):
    separated_priors.append(test[:,step:step+anchor_size_per_source[i],:])
    #torch.cat((separated_priors,test[:, step:step + anchor_size_per_source[i],:),0)
    step+=anchor_size_per_source[i]
    #buffer=test[:,step:step+separated_priors(i),4]
    #buffer=torch.Tensor(buffer)
    #print(buffer.size())
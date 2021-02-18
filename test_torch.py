'''
Author      : PureWhite
Date        : 2021-01-27 18:14:58
LastEditors : PureWhite
LastEditTime: 2021-02-18 17:45:54
Description : 
'''
import torch
flag = torch.cuda.is_available()
print(flag)
 
ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda()) 
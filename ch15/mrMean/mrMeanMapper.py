#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2011-02-21 15:02:50
LastEditor: John
LastEditTime: 2020-08-12 17:03:53
Discription: 
Environment: python3.7.7
'''

import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()
        
input = read_input(sys.stdin)#creates a list of input lines
input = [float(line) for line in input] #overwrite with floats
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

#output size, mean, mean(square values)
print("%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)))#calc mean of columns
print("report: still alive",file=sys.stderr)

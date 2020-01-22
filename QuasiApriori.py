#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:45:39 2020

@author: crowbe

Quasi Apriori data mining algorithm 

- Inspired by the Apriori data mining algorithm.
- Its goal is to analyze a boolean dataset and creates pairs of items. 
   I.e (Item1, Item2) - (Item3, Item4) - (Item5, Item6).
- A support threshold is calculated to have a minimum reference for a pair to be considered as useful information.

Video of the algorithm: https://youtu.be/7B9lXzsDmP0

"""

import numpy as np 
import pandas as pd 
from collections import Counter

# Loading the Data 
data = pd.read_excel('Market.xlsx') 
data.head()

#print data

d0 = data.iloc[:, 0].values #indicator_0 -->Item_1
d1 = data.iloc[:, 1].values #indicator_1 -->Item_2
d2 = data.iloc[:, 2].values #indicator_2 -->Item_3
d3 = data.iloc[:, 3].values #indicator_3- ->Item_4
d4 = data.iloc[:, 4].values #indicator_4 -->Item_5
d5 = data.iloc[:, 5].values #indicator_5 -->Item_6

# Creating the dataset's array
dataset = np.column_stack((d0,d1,d2,d3,d4,d5))

#print dataset

#print dataset.shape

transactions = dataset.shape[0]

frequencies = np.array([d0.sum(),d1.sum(),d2.sum(),d3.sum(),
                         d4.sum(),d5.sum()])

support_threshold = int(frequencies.sum()/d0.sum())

print ("support threshold:",support_threshold)

i = 0
freq_0 = [] 

for i in range(transactions-1): 
    
    if ((dataset[i,0],dataset[i,1])) == ((dataset[i+1,0],dataset[i+1,1])):
        #Puts a pair of items into a list
        freq_0.append((dataset[i,0],dataset[i,1]))
        
#print freq_0
#Counts unique pairs of the list.
print "column 0:"
print Counter(freq_0).keys()
print Counter(freq_0).values()

i = 0
freq_1 = [] 

for i in range(transactions-1): 
    
    if ((dataset[i,2],dataset[i,3])) == ((dataset[i+1,2],dataset[i+1,3])):
        freq_1.append((dataset[i,2],dataset[i,3]))

print "column 1:"       
print Counter(freq_1).keys()
print Counter(freq_1).values()

i = 0
freq_2 = [] 

for i in range(transactions-1): 
    
    if ((dataset[i,4],dataset[i,5])) == ((dataset[i+1,4],dataset[i+1,5])):
        freq_2.append((dataset[i,4],dataset[i,5]))

print "column 2:"       
print Counter(freq_2).keys()
print Counter(freq_2).values()
    

 









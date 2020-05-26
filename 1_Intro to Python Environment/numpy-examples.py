#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue Feb 20 22:48:34 2018

#@author: bp

# create a basic numpy array
import numpy as np
x = np.array([2, 4, 6, 8, 10])
print(x)
print(x[3])

#see what happens when you try to "add" lists
#try to divide as well and see what you get
x1 = [1,2,3,4,5]
print(type(x1))
x2 = [1,2,3,4,5]
print(x1 + x2)

# now define two numpy arrays - arange will automativally create 
# values from 0 to 19 for each array
# voila, all math operations in arrays work great! try division as well
y1 = np.arange(20)
y2 = np.arange(20)
print(y1)
print(y1+y2)
print(y1*y2)

# if you have a numpy array you can even derive conditional arrays 

print(y1 > 15)
print( y1[y1 > 15] )

# you can examine the number of dimensions and data types of numpy arrays

print(y1.ndim) # one dimension
print(y1.dtype) # integer array 64 bits

# two-dimensional arrays
y3 = np.array([[1,2],[4,5]])
print(y3)
# query size, shape of arrays and quick operations
print(y1.size, y1.shape, y1.sum(), y3.size, y3.shape, y3.sum())

#pandas uses numpy to represent enter data frames, like a spreadsheet, or a dataset

# example below from Google Cloud Platform's "Wrangling Data with Pandas"

#download iris.csv file with the row headers from the Web
import pandas as pd
iris_csv = pd.read_csv("iris.csv")
print(type(iris_csv))

# head and describe return top few rows and summary stats
print(iris_csv.head())
print(iris_csv.describe())

#use dot columns method on the data frame to get the column names
#use brackets notation to pull out a column
print(iris_csv.columns)
print(iris_csv['sepal_length'])
# to retrive a row of data use index location (iloc) function with row number
print(iris_csv.iloc[0])
#accessing a row and column
print(iris_csv['sepal_length'].iloc[0])

#to access a subset of rows and columns, first create names of colums
cols = iris_csv[['sepal_length', 'sepal_width']]
print(cols.iloc[5:20])

#also go through basic pandas data frame tutorial below
#https://github.com/adeshpande3/Pandas-Tutorial/blob/master/Pandas%20Tutorial.ipynb



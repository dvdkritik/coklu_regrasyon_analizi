import pandas as pd
import numpy as np

count = 0
#Read CSV file 

def readfromcsv():

    rf = pd.read_csv('csvfile.csv', index_col = False, header = 0)
    #print rf
    #print rf.values

    rf = rf.fillna(0)
    #print rf

    #print rf.head()  #prints first five rows of dataset
    #print rf.describe() #prints mean, min, max, count of dataset


##for i in range(0,len(rf.values)):
##    for j in range(0,len(rf.values[0])):
##        if np.isnan(rf.values[i][j]) :
##            count = count +1
##            rf.values[i][j] = 0.0
##print count
##print rf.values
##data = np.array(rf)
##print data
##
##for i in range(0,len(data)):
##    for j in range(0,len(data[0])):
##        if data[i][j]. :
##            count = count +1
##            data[i][j] = 0
##print data
##            

def pdseries():
    s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

    #retrieve the first element
    print s[0],s[4]     #print 0th and 4th index element 
    print s             #print series
    print s[['a','c']]  #print value a & c with index
    #print s['g']        #key error

def dataframe():
    data = [['Alex',10],['Bob',12],['Clarke',13]]
    #df = pd.DataFrame(data,columns=['Name','Age']) #df = data ['Name'-> column,'Alex'-> Row1,'Bob'->Row2,'Clarke'->Row3] 
    #or
##    data = {'Age':[10,12,13],'Name':['Alex','Bob','Clarke']}
##    df = pd.DataFrame(data)
##    print df
    df = pd.DataFrame(data,columns=['Name','Age'],dtype=float) #data type float
    #df = pd.DataFrame(data, index=['rank1','rank2','rank3']) #Indexed data frames
    print data
    print df
dataframe()

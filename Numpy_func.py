import numpy as np
import datetime
## array [x:y] where x is index from which array needs to start and y is till
## when array is present



# Create numpy single dimention array 
def create_1_dim_array():
    a = np.array([1, 2, 3])   # Create a rank 1 array
    #print(type(a))            # Prints "<class 'numpy.ndarray'>"
    #print(a.shape)            # Prints "(3,)"
    #call array by element same as array
    print a[0], a[1], a[2]    # Prints "1 2 3"
    #change array element by index 
    a[0] = 5                  # Change an element of the array
    print a                   # Prints "[5, 2, 3]"

def create_2_dim_array():
    b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
    #print(b.shape)                     # Prints "(2, 3)"
    #print b
    #array indexing
    print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"

def create_blank_arrays(n,m,k):
    #n*m array with K value if K is not used shoulb be left 0
    a = np.zeros((n,m))  #create zeros of n*m
    #a = np.ones((n,m)) #create ones of n*m
    #a = np.full((n,m),k) #create k element of n*m
    #a = np.empty((n,m))  #create empty of n*m
    print a

def create_multi_dim_array():
    a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    print a
    # array [ limit rows, limit columns]
    # Cropping of array
    subarr = a[:2, 1:3]
    #print subarr

    # Accessing a row of array
    row = a[1]    # or c[1,:]
    #print row

    # Accessing a column
    column = a[: , 1]
    #print column

    # Element of array array[row, cloumn]
    #print(a[0, 1])   # Prints "2"
    # change element of array
    #a[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
    #print(a[0, 1])   # Prints "77"

    # Integer indexing arr([[row numbers],[column element of perticular row]] 
    #print a[[0, 1,2], [0, 1,3]] #0,1,2 rows (element 0 of 0th row,1st of 1st row
    #3rd of 2nd row
    # same can be achived by below trick - This trick can be useful while there is
    # need to get element/to operate on one element of each row by column index
    #b = [0, 1, 3]
    #print a [np.arange(3),b]
    #a [np.arange(3),b] += 10
    #print a
    # equivalant to [a[0,0],a[1,1]]
    #print a[[0,0,0],[1,2,3]] # will print oth row 1,2,3 elements which is 2,3,4

    # bool index is used to operate on entire array
    #bool_index = (a>2)
    #print bool_index

def speed_check():
    a = np.random.random_integers(5, high=6, size=(10,10))
    ##print a
    ##curr_time = datetime.datetime.now()
    ##for i in range(0,len(a)):
    ##    for j in range(0,len(a[0])):
    ##        if a[i][j] > 5 :
    ##            a [i][j] = True
    ##        else:
    ##            a[i][j] = False

    ##post_time = datetime.datetime.now()
    ##bool_index = (a>5)
    ##
    ##diff = post_time - curr_time
    ##print diff
    ##print bool_index

def array_operation():
    # dtype is datatype which can be int64,float64 etc 
    x = np.array([[1,2],[3,4]], dtype=np.float64)
    y = np.array([[5,6],[7,8]], dtype=np.float64)

    # Elementwise sum; both produce the array
    # [[ 6.0  8.0]
    #  [10.0 12.0]]
    #print(x + y) #same as print(np.add(x, y))

    # Elementwise difference; both produce the array
    # [[-4.0 -4.0]
    #  [-4.0 -4.0]]
    #print(x - y) #same as print(np.subtract(x, y))

    # Elementwise product; both produce the array is done by '*'
    # [[ 5.0 12.0]
    #  [21.0 32.0]]
    #print(x * y) #same as print(np.multiply(x, y))

    #  Product of array (dot product) is done by
    #print x.dot(y)

    

    # Elementwise division; both produce the array
    # [[ 0.2         0.33333333]
    #  [ 0.42857143  0.5       ]]
    #print(x / y) #same as print(np.divide(x, y))

    # Elementwise square root; produces the array
    # [[ 1.          1.41421356]
    #  [ 1.73205081  2.        ]]
    #print(np.sqrt(x))

    # Numpy provides sum - row wise / column wise / whole matrix
    #print np.sum(x) # matrix sum
    #print np.sum(x, axis=0) # row wise
    #print np.sum(x, axis=1) # column wise

    # Transpose is done by
    #print x.T
    
    # Stacking of array
    #print np.hstack((x,y)) #Horizontal stacking
    #print np.vstack((x,y)) #Vertical stacking


    # Spliting of array
    a = np.floor(10*np.random.random((2,12)))
    #print np.hsplit(a,3)   # Split a into 3
    #print np.hsplit(a,(3,4)) #Split after 3rd and 4th column

def broadcasting():
    x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
    y = np.empty_like(x)   # Create an empty matrix with the same shape as x

    v = np.array([1,2,1])
    #for i in range(0,len(x)):
    #    y[i,:] = x[i,:] + v
    # Same can be done by
    #vv = np.tile(v,(4,1)) # Will copy v 4 times horizontally and 1 time verticaly
    #y = x + v
    # Or Just
    #y = x + v
    #print y




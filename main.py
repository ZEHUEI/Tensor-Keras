import tensorflow as tf
import numpy as np

# print(tf.__version__)

scalar = tf.constant(7)
# print(scalar)
tf.print(scalar)
########################################################################################
# print(scalar.ndim)
########################################################################################
vector = tf.constant([10,10])
# print(vector)
########################################################################################
# print(vector.ndim)
########################################################################################
# #matrix
matrix = tf.constant([[10,7],[7,10]])
# print(matrix)
# print(matrix.ndim)
########################################################################################
another_matrix = tf.constant([[10.,7.],[3.,2.],[8.,9.]],dtype=tf.float16)
# print(another_matrix)
# print(another_matrix.ndim)

tensor = tf.constant([[[1,2,3],
                       [4,5,6]],
                       [[7,8,9],
                       [10,11,12]],
                       [[13,14,15],
                      [16,17,18]]])
# print(tensor)
# print(tensor.ndim)

# What we have created so far:
# Vector: a number with direction (e.g) wind speed and direction)
# Matrix: a 2D array of numbers
# Tensor: an n-dimensional array of numbers (when n can be any number, a 0 dimensional tensor is a scalr and 1 dimensional tensor is a vector)
########################################################################################

changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])
# print(changeable_tensor,unchangeable_tensor)

#we change one of the element in our changable tensor
# so we chang 10 to 0

#this is err
# changeable_tensor[0] = 7
# print(changeable_tensor)

#we need to use assign
changeable_tensor[0].assign(7)
# print(changeable_tensor)

#u cannot change constant
# unchangeable_tensor[0].assign(7)
# print(unchangeable_tensor)

##############################################################
#creating random tensors
#random tensors are tensors of some abitary size which contains random numbers

#create 2 random tensors
#normal distribution

random_1 = tf.random.Generator.from_seed(42) #set seed for reproducibility
random_1 = random_1.normal(shape=(3,2))
# print(random_1)

random_2=tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))
# print(random_2)

#are they equal?
random_1, random_2, random_1 == random_2
# print(random_1==random_2)
# print(tf.reduce_all(random_1==random_2))

######################################################
#shuffle the order of tensor, train test split, cross validation
not_shuffled = tf.constant([[10,7],
                            [3,4],
                            [2,5]])
# print(not_shuffled)
#this make it the same
###
tf.random.set_seed(42)
not_shuffled = tf.random.shuffle(not_shuffled,seed=42)
###
# print(not_shuffled)

#exercise https://www.tensorflow.org/api_docs/python/tf/random/set_seed
#create 5 diff tensor with tf.constant then shuffle

###############################################################
#creating tensors from numpy arrays
haha = tf.ones([6,7])#rows then column
bruh = tf.zeros(shape=(3,4))

#turn Numpy arrays into TensorFlow tensors
#tensor can run on a GPU much faster
numpy_A = np.arange(1,25,dtype=np.int32)
# 2 dimension, 3 rows, 4columns have to be the same elements in the 1-25 which is 24 range if not error
A = tf.constant(numpy_A,shape=(2,3,4))
B = tf.constant(numpy_A,shape=(3,8))
# print(A,"\n\n",B)
##################################################################

#getting info from tensors
#Shape
#Rank
#Axis or dimension
#Size

#4 dimension tensor
# this is 2 rows then 3 elements then 3 rows then 5 elements
rank_4_tensor = tf.zeros(shape=(2,3,4,5))
# print(rank_4_tensor)

#makes 2 to null
# print(rank_4_tensor[0])

# print(rank_4_tensor.shape,rank_4_tensor.ndim,tf.size(rank_4_tensor))

#Get various attributes of our tensor
# rank_4_tensor = tf.zeros(shape=(2,3,4,5))
# print("Datatype of every element:", rank_4_tensor.dtype)
# print("No. of dimensions (rank):",rank_4_tensor.ndim)
# print("shape of tensor:",rank_4_tensor.shape)
# print("elements along of 0 axis:",rank_4_tensor.shape[0])
# print("elements along the last axis:",rank_4_tensor.shape[-1])
# print("total No. of elements in our tensor",tf.size(rank_4_tensor).numpy())


##########################################################################################
#indexing and expanding tensors
#Get first 2 elements of each dimensions
some_list = [1,2,3,4]
# print(some_list[:2])
# print(rank_4_tensor[:2,:2,:2,:2])

#get first element from dimension from each index except from final one
# print(some_list[:1])
# print(rank_4_tensor)
# print(rank_4_tensor[:1,:1,:1,:])

#Create a rank 2 tensor (2D)
rank_2_tensor = tf.constant([[6,7],[6,7]])
# print(rank_2_tensor.ndim)
#Get the last item of each of rank 2 tensor
# print(rank_2_tensor[:,-1])

#adding another dimension
#[...,xxx] ... means [:,:,tf.newaxis] if big dimension and want to add behing just ...
# rank_3_tensor = rank_2_tensor[...,tf.newaxis]
# print(rank_3_tensor.ndim)
#or use this: -1 add dimension at the back
tf.expand_dims(rank_2_tensor,axis=-1)
# print(tf.expand_dims(rank_2_tensor,axis=-1))
#
# print(tf.expand_dims(rank_2_tensor,axis=0))

###################################################################
#Manipulating tensors (tensor operations) with basic operations
tensor = tf.constant([[10,7],[3,4]])
# print(tensor+10)
#this is faster
# tensor = tf.multiply(tensor,10)
# print(tensor)
# print(tensor-10)
# print(tensor/10)

##################################################################
#matrix multiplication with tensors P1
tf.matmul(tensor,tensor)
A = tf.constant([[1,2,5],[7,2,1],[3,3,3]])
B = tf.constant([[3,5],[6,7],[1,8]])
# print(A,"\n\n",B)
# print(tf.matmul(A,B))
# print(tf.matmul(tensor,tensor))
# print(tensor*tensor)

#Matrix multiplication with python operator "@"
# print(A@B)
################################################################
#Part 2
X = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
Y = tf.constant([[7,8],
                 [9,10],
                 [11,12]])
# print(tf.matmul(tf.reshape(X,shape=(2,3)),Y))

# Y = tf.reshape(Y,shape=(2,3))

# print(X,Y)
# print(X@Y)
#
# ANS = tf.matmul(X,Y)
# print(ANS)

#can do the same with transpose
#makes 3x2 to 2x3
ng = tf.transpose(X)
# print(ng)

mg = tf.matmul(tf.transpose(X),Y)
# print(mg)

###################################################################
#Part 3
#perform the dot product on X and Y which X or Y to be transposed
X = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
Y = tf.constant([[7,8],
                 [9,10],
                 [11,12]])

# print(tf.tensordot(tf.transpose(X),Y,axes=1))
################################################################
#perform matrix multiplication between X and Y (transposed)
# print(tf.matmul(X,tf.transpose(Y)))

#perform matrix multiplication between X and Y 9reshaped)
# print(tf.matmul(X,tf.reshape(Y,shape=(2,3))))

#check the values of Y, reshape Y and transposed Y
print("Normal Y:")
print(Y,'\n')
print("Y reshaped to (2,3):")
print(tf.reshape(Y,shape=(2,3)),'\n')
print("Y transposed:")
print(tf.transpose(Y))

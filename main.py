import tensorflow as tf

print(tf.__version__)

scalar = tf.constant(7)
print(scalar)
tf.print(scalar)

print(scalar.ndim)

vector = tf.constant([10,10])
print(vector)

print(vector.ndim)

#matrix
matrix = tf.constant([[10,7],[7,10]])
print(matrix)
print(matrix.ndim)

another_matrix = tf.constant([[10.,7.],[3.,2.],[8.,9.]],dtype=tf.float16)
print(another_matrix)
print(another_matrix.ndim)
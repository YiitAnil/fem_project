#!/usr/bin/env python3
# Main - 1D elastic bar - Linear FEM

# Prepare environment and import libraries

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=14)
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Multiply, Add, Lambda, Dense, Dot, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint

from pinn.layers import getScalingDenseLayer


def build_mlp(dLInputScaling, mlp_name):
        model = Sequential([
#            dLInputScaling,
            Dense(5,activation = 'tanh'),
#            Dense(10,activation = 'elu'),
#            Dense(20,activation = 'elu'),
#            Dense(40,activation = 'elu'),
            Dense(64,activation = 'sigmoid')
            ], name=mlp_name)
        optimizer = RMSprop(1e-2)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model
    
    
class AMatrix(Layer):
    """        
    Elastic stiffness matrix
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMatrix, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape = [1,8,8],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      constraint = self.kernel_constraint,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
        output = self.kernel
#        output = array_ops.reshape(output,(array_ops.shape(output)[0],1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 

class FMatrix(Layer):
    """        
    Force matrix
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMatrix, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.kernel = self.add_weight("kernel",
                                      shape = [8],
                                      initializer = self.kernel_initializer,
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      constraint = self.kernel_constraint,
                                      **kwargs)
        self.built = True

    def call(self, inputs):
#        kernel_enhanced = array_ops.tile(tf.transpose([self.kernel]), tf.constant([1,array_ops.shape(inputs)[0]]))
#        kernel_enhanced = tf.expand_dims(self.kernel,0)
        output = self.kernel * inputs
#        output = array_ops.reshape(output,(array_ops.shape(output)[0],1))
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 

def create_fe_model(delta_stiffness_mlp, elastic_stiffness, force_vector, 
                    stiffness_low, stiffness_up, batch_input_shape, myDtype):
    
    inputLayer = Input(shape=(1,))   
    
    elasticStiffnessLayer = AMatrix(input_shape = inputLayer.shape, dtype = myDtype, trainable=False)
    elasticStiffnessLayer.build(input_shape = inputLayer.shape)
    elasticStiffnessLayer.set_weights([np.asarray(elastic_stiffness, dtype = elasticStiffnessLayer.dtype)]) 
    elasticStiffnessLayer = elasticStiffnessLayer(inputLayer)
    
    forceMatrixLayer = FMatrix(input_shape = inputLayer.shape, dtype = myDtype, trainable=False)
    forceMatrixLayer.build(input_shape = inputLayer.shape)
    forceMatrixLayer.set_weights([np.asarray(force_vector, dtype = forceMatrixLayer.dtype)]) 
    forceMatrixLayer = forceMatrixLayer(inputLayer)
    
    deltaStiffnessLayer = delta_stiffness_mlp(inputLayer)
    scaledDeltaStiffnessLayer = Lambda(lambda x, stiffness_low=stiffness_low, stiffness_up=stiffness_up:
    x*(stiffness_up-stiffness_low)+stiffness_low)(deltaStiffnessLayer)
    
    deltaStiffnessReshapedLayer = Reshape((8, 8))(scaledDeltaStiffnessLayer)
    
    correctedStiffnessLayer = Multiply()([elasticStiffnessLayer, deltaStiffnessReshapedLayer])
    
    inverseStiffnessLayer = Lambda(lambda x: tf.linalg.inv(x))(correctedStiffnessLayer)
    
#    deflectionOutputLayer = Lambda(lambda x: tf.linalg.matmul(x[0],x[1]))([inverseStiffnessLayer, forceMatrixLayer])
    deflectionOutputLayer = Dot((1))([inverseStiffnessLayer, forceMatrixLayer])
    
    functionalModel = Model(inputs = [inputLayer], outputs = [deflectionOutputLayer])
    
    functionalModel.compile(loss=mse,
                  optimizer=RMSprop(5e-3),
                  metrics=[mae])
    return functionalModel


def create_physics_model(elastic_stiffness, force_vector, 
                    stiffness_low, stiffness_up, batch_input_shape, myDtype):
    
    inputLayer = Input(shape=(1,))   
    
    elasticStiffnessLayer = AMatrix(input_shape = inputLayer.shape, dtype = myDtype, trainable=False)
    elasticStiffnessLayer.build(input_shape = inputLayer.shape)
    elasticStiffnessLayer.set_weights([np.asarray(elastic_stiffness, dtype = elasticStiffnessLayer.dtype)]) 
    elasticStiffnessLayer = elasticStiffnessLayer(inputLayer)
    
    forceMatrixLayer = FMatrix(input_shape = inputLayer.shape, dtype = myDtype, trainable=False)
    forceMatrixLayer.build(input_shape = inputLayer.shape)
    forceMatrixLayer.set_weights([np.asarray(force_vector, dtype = forceMatrixLayer.dtype)]) 
    forceMatrixLayer = forceMatrixLayer(inputLayer)
       
    inverseStiffnessLayer = Lambda(lambda x: tf.linalg.inv(x))(elasticStiffnessLayer)
    
#    deflectionOutputLayer = Lambda(lambda x: tf.linalg.matmul(x[0],x[1]))([inverseStiffnessLayer, forceMatrixLayer])
    deflectionOutputLayer = Dot((1))([inverseStiffnessLayer, forceMatrixLayer])
    
    functionalModel = Model(inputs = [inputLayer], outputs = [deflectionOutputLayer])
    
    functionalModel.compile(loss=mae,
                  optimizer=RMSprop(5e-3),
                  metrics=[mse])
    return functionalModel


# --------------------------
# Functions definition
def Mesh1D(L1, Nx):
    # Generates nodes positions and connectivity table for 1D mesh of length L1 
    # and number of elements Nx
    # Linear elements only
    # Nodes array contains nodal positions (one node per row)
    # Connectivity array contains the element nodes number (one element per each row)
    
    
    # TODO
    Nodes = np.linspace(0,L1,Nx+1)
    Connectivity = np.zeros(shape=(Nx, 2), dtype = 'int')
    for e in range(0, Nx):
        Connectivity[e,0] = e
        Connectivity[e,1] = e+1
        
    return Nodes, Connectivity
        


def LinElement1D(Nodes_el, EA, q):
    # Generates load vector and stiffness matrix at the element level
    
    
    # TODO
    K_el = EA / (Nodes_el[1] - Nodes_el[0]) * np.array([[1,-1],[-1,1]])
    q_el = (q * (Nodes_el[1] - Nodes_el[0]) / 2) * np.transpose(np.array([1,1]))
    
    return K_el, q_el
 
    

# 
# --------------------------
# MAIN

#
# Input ------------------------------------------------------
#

L1 = 200.0    # Lengh of elastic bar
Nx = 8      # Number of elements

# Material Properties
EA = np.ones(shape=(Nx, 1))     
for i in range(0, Nx):      # Modify this loop to assign different material properties per element
    EA[i,0] = 73084*100
    
# EBC
EBC = np.array([0, 0], dtype='int')    # Assign EBC in the form [dof, dof value]
   
# Distributed loads and NBC
q = 0           # Distributed load (assumed constant)
NBC = [Nx, 40000]   # Assign NBC in the form [dof, load value]
    
    
#
# Meshing ----------------------------------------------------
#

Nodes, Connectivity = Mesh1D(L1, Nx)

#
# Element calculations and assembly --------------------------
#

K_model = np.zeros(shape=(Nx+1, Nx+1))
f_model = np.zeros(shape=(Nx+1, 1))
for e in range(0, Nx):
    # TODO
    Nodes_el = Connectivity[e]
    Nodes_loc = np.array([Nodes[Nodes_el[0]],Nodes[Nodes_el[1]]])
    K_el, q_el = LinElement1D(Nodes_loc, EA[e], q)
    
    K_model[Nodes_el[0], Nodes_el[0]] += K_el[0,0]
    K_model[Nodes_el[0], Nodes_el[1]] += K_el[0,1]
    K_model[Nodes_el[1], Nodes_el[0]] += K_el[1,0]
    K_model[Nodes_el[1], Nodes_el[1]] += K_el[1,1]
    
    f_model[Nodes_el[0]] += q_el[0]
    f_model[Nodes_el[1]] += q_el[1]
    
    if e == NBC[0]-1:
        f_model[Nodes_el[1]] += NBC[1]
    
    
#
# Apply element EBC --------------------------
#      
    
    # TODO
A_matrix = K_model[EBC[0]+1:,EBC[0]+1:]
B_matrix = K_model[EBC[0]+1:,0]
C_matrix = K_model[EBC[0],0]

F_matrix = f_model[EBC[0]+1:,0]

#
# Solve for displacements and reaction forces - plot solution ----------------
# 

    # TODO
u =  np.linalg.solve(A_matrix,F_matrix-np.transpose(B_matrix*EBC[1]))
R = np.dot(B_matrix,u) + C_matrix * EBC[1]

#fig = plt.figure()
#plt.plot(Nodes, np.zeros(Nx+1),'k-', linewidth = 10,label = 'Bar')
#plt.plot(Nodes, np.append(np.array(EBC[0]),u),'r-o', linewidth = 3,label = 'Solution')
#plt.xlabel('x (mm)')
#plt.ylabel('u (mm)')
#plt.xlim(0,L1)
#plt.xticks(Nodes)
#plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()


plastic_io = pd.read_csv('./plastic_deflections.csv', index_col = False, header = None)

force_input = np.asarray(plastic_io)[0,:]
plastic_deflections = np.asarray(plastic_io)[1:,:]
force_elastic = np.array([40000.0])


force_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

physics_model = create_physics_model(np.array([A_matrix]), force_vector, 
                    2e5, 6e5, force_input.shape, 'float32')

elastic_deflection = physics_model.predict(force_elastic)
dLInputScaling = getScalingDenseLayer(np.array([force_input.min(axis=0)]), np.array([force_input.max(axis=0)-force_input.min(axis=0)]))

delta_stiffness_mlp = build_mlp(dLInputScaling, 'delta_stiffness')
delta_stiffness_mlp.trainable = True


fe_model = create_fe_model(delta_stiffness_mlp, np.array([A_matrix]), force_vector, 
                           -1e0, 1e0, force_input.shape, 'float32')


weight_path = "./1d_linear_bar_model_test6/cp.ckpt"

ModelCP = ModelCheckpoint(filepath=weight_path, monitor='loss',
                                                     verbose=1, save_best_only=True,
                                                     mode='min', save_weights_only=True) 
ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.85,
                                   min_lr = 1e-15, patience=1000, verbose=1, mode='min')
ToNaN = TerminateOnNaN()
callbacks_list = [ReduceLR, ToNaN, ModelCP]
EPOCHS = 20000

history = fe_model.fit(force_input, np.transpose(plastic_deflections), epochs=EPOCHS, verbose=1, callbacks=callbacks_list)

prediction = fe_model.predict(force_input)

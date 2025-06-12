import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, InputLayer, Input, add, dot, maximum, average, multiply
from tensorflow.keras.layers import concatenate, Reshape, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd

def circulant(shape):
    matrix_shape = shape.get_shape().as_list()
    circulant_set = []
    circulant_ = K.expand_dims(shape, 2)
    circulant_set.append(circulant_)

    for i in range(1, matrix_shape[1]):
        pre=circulant_[:,(matrix_shape[1]-i):,:]
        host=circulant_[:,0:(matrix_shape[1]-i),:]
        circulant_1=tf.concat([pre,host],1)
        circulant_set.append(circulant_1)

    vector_ = K.concatenate(circulant_set)

    return vector_

class Circulant_layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Circulant_layer, self).__init__(**kwargs)
   
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Circulant_layer, self).build(input_shape) 

    def call(self, x):
        assert isinstance(x, list)
        des, body = x
        
        des = K.dot(des, self.kernel) 
        body = K.dot(body, self.kernel)
        
        multiple_vector1 = K.batch_dot(circulant(body), des)
        multiple_vector2 = K.batch_dot(circulant(des), body)
        
        sum_vector = multiple_vector1 + multiple_vector2
        sum_vector_1 = K.dot(sum_vector, self.kernel)

        return sum_vector
    
class network:
    def __init__(self, des_shape, body_shape, hid_layer, output_layer):
        self.des_shape = des_shape
        self.body_shape = body_shape
        self.hid_layer = hid_layer  
        self.output_layer = output_layer
        
    ''' OPCNN '''    
    def OPCNN_32(self):
        # des modality
        des_input = Input(shape = (self.des_shape)) 
        body_input = Input(shape = (self.body_shape))
        
        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)
        
        des_reshape_2 = Reshape((1,self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1,self.hid_layer[0]))(body_fc)
        
        dot_layer = dot([des_reshape_2, body_reshape_2], axes = 1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)
        
        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)      
        res_ab = res_a + res_a_2
        
        res_b = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_b_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b)
        res_b_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b_1)
        res_bc = res_b + res_b_2
        
        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_bc)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1] , activation = 'relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2] , activation = 'relu')(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.output_layer)(x)
        
        OPCNN = keras.models.Model(inputs = [des_input, body_input], outputs = [x])  

        return OPCNN
    
    ''' OPCNN '''    
    def OPCNN_31(self):
        # des modality
        des_input = Input(shape = (self.des_shape)) 
        body_input = Input(shape = (self.body_shape))
        
        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)
        
        des_reshape_2 = Reshape((1,self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1,self.hid_layer[0]))(body_fc)
        
        dot_layer = dot([des_reshape_2, body_reshape_2], axes = 1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)
        
        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)      
        res_ab = res_a + res_a_2
        
        res_b = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_b_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b)
        res_b_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b_1)
        res_bc = res_b + res_b_2
        
        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_bc)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1] , activation = 'relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)
        
        OPCNN = keras.models.Model(inputs = [des_input, body_input], outputs = [x])  

        return OPCNN

    ''' OPCNN '''

    def OPCNN_22(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2], activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    def OPCNN_21(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    def OPCNN_12(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        tensor_flat = Flatten()(res_ab)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2], activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    def OPCNN_11(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        tensor_flat = Flatten()(res_ab)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    ''' OPCNN '''

    def OPCNN_32_Drop(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        res_b = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_b_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b)
        res_b_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b_1)
        res_bc = res_b + res_b_2

        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_bc)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2], activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    ''' OPCNN '''

    def OPCNN_31_Drop(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        res_b = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_b_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b)
        res_b_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_b_1)
        res_bc = res_b + res_b_2

        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_bc)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    ''' OPCNN '''

    def OPCNN_22_Drop(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2], activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    def OPCNN_21_Drop(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        res_c = Conv2D(32, (3, 3), activation='relu', padding="same")(res_ab)
        res_c_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c)
        res_c_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_c_1)
        res_cc = res_c + res_c_2

        tensor_flat = Flatten()(res_cc)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    def OPCNN_12_Drop(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        tensor_flat = Flatten()(res_ab)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.hid_layer[2], activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

    def OPCNN_11_Drop(self):
        # des modality
        des_input = Input(shape=(self.des_shape))
        body_input = Input(shape=(self.body_shape))

        des_fc = Dense(self.hid_layer[0])(des_input)
        body_fc = Dense(self.hid_layer[0])(body_input)

        des_reshape_2 = Reshape((1, self.hid_layer[0]))(des_fc)
        body_reshape_2 = Reshape((1, self.hid_layer[0]))(body_fc)

        dot_layer = dot([des_reshape_2, body_reshape_2], axes=1, normalize=False)
        dot_reshape = Reshape((self.hid_layer[0], self.hid_layer[0], 1))(dot_layer)

        res_a = Conv2D(32, (3, 3), activation='relu', padding="same")(dot_reshape)
        res_a_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a)
        res_a_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(res_a_1)
        res_ab = res_a + res_a_2

        tensor_flat = Flatten()(res_ab)

        x = Dense(self.hid_layer[1], activation='relu')(tensor_flat)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)

        x = Dense(self.output_layer)(x)

        OPCNN = keras.models.Model(inputs=[des_input, body_input], outputs=[x])

        return OPCNN

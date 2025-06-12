from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sm = SMOTE(random_state=202004)

from keras import backend as K

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def NN1_loss(y_true,y_pred):
  mu = tf.slice(y_pred,[0,0],[-1,1])
  sigma = tf.math.exp(tf.slice(y_pred,[0,1],[-1,1]))     
  loss = tf.reduce_sum(tf.square(y_true - mu)/sigma + tf.math.log(sigma),axis=0)
  return loss 


class CV_Test:
    def __init__(self, X_train, y_train, X_test, y_test, model, tfn=None, seed_num=0, model_name='model_name'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

        self.des_train = X_train[:,:2048]
        self.body_train = X_train[:,2048:]
        self.des_test = X_test[:,:2048]
        self.body_test = X_test[:,2048:]
        self.TFL = tfn
        self.seed_num = seed_num
        self.model_name = model_name

    ''' None '''
    def base(self):
        base_y = dict()

        
        des_dev, des_test = self.des_train, self.des_test
        body_dev, body_test = self.body_train, self.body_test
        y_dev, y_test = self.y_train, self.y_test
        
        checkpoint_filepath = 'D:/checkpoint/seed_{}/'.format(self.seed_num)

        best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={"NN1_loss": NN1_loss})

  
        if self.TFL is None:
            y_pred = best_model.predict([des_test[:], body_test[:]])

        else:
            y_pred = best_model.predict([np.ones(des_test.shape[0]), des_test[:], body_test[:]])

        base_y = y_pred

        return base_y
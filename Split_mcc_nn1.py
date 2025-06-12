from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import StandardScaler

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def NN1_loss(y_true, y_pred):
  mu = tf.slice(y_pred, [0,0], [-1,1])
  sigma = tf.math.exp(tf.slice(y_pred, [0,1], [-1,1]))     
  loss = tf.reduce_sum(tf.square(y_true - mu)/sigma + tf.math.log(sigma),axis=0)
  return loss 

sm = SMOTE(random_state=202004)


class CV_Train:
    def __init__(self, X_train, y_train, X_test, y_test, model, epochs, learning_rate, tfn=None, seed_num=0, model_name='model_name'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
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

        model_temp = tf.keras.models.clone_model(self.model)

        
        train_index, val_index = train_test_split(range(len(self.X_train)), test_size = 0.2,
                                                   random_state = self.seed_num, shuffle = True)
        
        
        des_train, des_val = self.des_train[train_index], self.des_train[val_index]
        body_train, body_val = self.body_train[train_index], self.body_train[val_index]
        y_train, y_val = self.y_train[train_index], self.y_train[val_index]

        des_test, body_test = self.des_test, self.body_test
        y_test = self.y_test
        
        checkpoint_filepath = 'D:/checkpoint/seed_{}/'.format(self.seed_num)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_mean_squared_error',
            mode='min',
            save_best_only=True)

        model_temp.compile(optimizer= keras.optimizers.Adam(learning_rate = self.learning_rate),
                           loss=NN1_loss, metrics = ["mean_squared_error"])

        if self.TFL is None:
            kf_history = model_temp.fit(x = [des_train, body_train], y = y_train, epochs=self.epochs,
                                        validation_data=([des_val, body_val], y_val),
                           callbacks = [model_checkpoint_callback])

            best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={"NN1_loss": NN1_loss})
            y_pred = best_model.predict([des_test[:], body_test[:]])

        else:
            kf_history = model_temp.fit(x = [np.ones(des_train.shape[0]),des_train, body_train], y = y_train, epochs=self.epochs,
                                        validation_data=([np.ones(des_val.shape[0]),des_val, body_val], y_val),
                                        callbacks = [model_checkpoint_callback])

            best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={"NN1_loss": NN1_loss})
            y_pred = best_model.predict([np.ones(des_test.shape[0]), des_test[:], body_test[:]])

        base_y = y_pred

        return base_y
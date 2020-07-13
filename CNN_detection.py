import numpy as np
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#configuring keras
import keras
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GaussianDropout
from keras.utils import plot_model
import data_processing as dp
from keras import backend as K


# NOTE: comment the next line if running more than once back-to-back
dp.save_data('./dataset/data.csv', binary=1, augment=0)
eeg_data, labels = dp.load_data()

inputs = Input(shape=(178, 1, 1))
x = GaussianDropout(0.1)(inputs)
x = Conv2D(24, kernel_size=(5, 1), strides=3, activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3,1), strides=2, activation="relu")(x)
x = BatchNormalization()(x)
x = Conv2D(8, (3,1), strides=2, activation="relu")(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(20)(x)
x = Dropout(0.3)(x)
predictions = Dense(units=2, activation="softmax")(x)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = Model(inputs, predictions)
#plot_model(model, to_file='ANN_Conv2D_AutoAug__BinaryClass.png', show_shapes=True)
model.load_weights("ANN_Conv2D_AutoAug_BinaryClass.h5")

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])

csv_logger = keras.callbacks.CSVLogger("ANN_Conv2D_AutoAug__BinaryClass", separator=',', append=False)
es = keras.callbacks.EarlyStopping(patience=10)

#model.fit(eeg_data, labels, epochs=200, verbose=1, validation_split=0.1, callbacks=[es, csv_logger])
#model.save("ANN_Conv2D_AutoAug_BinaryClass.h5")
x_test = np.load("./dataset/x_test.npy")
y_test = np.load("./dataset/y_test.npy")
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test,y_test, verbose=0)
print(f"loss is {loss}")
print(f"accuracy is {accuracy}")
print(f"f1_score is {f1_score}")
print(f"precision is {precision}")
print(f"recall is {recall}")

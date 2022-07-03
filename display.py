import tensorflow as tf
import numpy as np
import os
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GaussianDropout
from keras.utils import plot_model
import tkinter as Tkin
top = Tkin.Tk()

top.configure(background = 'LightBlue')



class GUI:
	def gen_callback(self):
		self.a=np.random.randint(-150, 150, 178)
		print(self.a)
		plt.plot(self.a)
		plt.show()

	def test_callback(self):
		self.a = self.a.reshape((1,178,1,1))
		results = model.predict([self.a])
		results = np.argmax(results,axis=1)
		if(results==1):
			outputTxt.config(text="Most Likely to have a seizure",font=('Helvetica', '20'),fg="Brown",bg="LightGreen")
		else:
			outputTxt.config(text="Most Likely to not have a seizure",font=('Helvetica', '20'),fg="Brown",bg="LightGreen")

gui = GUI()
time = []
amplitude = []
gen = Tkin.Button(top,text="GENERATE",height=10, width=50,font=('Helvetica', '15'),command=gui.gen_callback,padx=70,pady=70,bg="blue", fg="white")
test =Tkin.Button(top,text="TEST",height=10, width=50,font=('Helvetica', '15'),command=gui.test_callback,padx=70,pady=70,bg="blue", fg="white")
outputTxt = Tkin.Label(top,text="");
outputTxt.pack()
gen.pack()
test.pack()

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
model = Model(inputs, predictions)

model.load_weights("C:/Users/admin/Desktop/study/tensorlflow/ANN_Conv2D_AutoAug_BinaryClass.h5")
top.geometry("1000x1000+300+300")

top.mainloop()

#a=np.expand_dims(a,axis=-1)

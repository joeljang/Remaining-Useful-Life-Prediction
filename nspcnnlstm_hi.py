import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Conv2D, BatchNormalization, Dropout, Flatten, TimeDistributed, LSTM
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

num_timesteps = 100 #Number of time steps

data_dir = sys.argv[1] #training data
data_dir2 = sys.argv[2] #testing data

#Loading Training Data
lst = os.listdir(data_dir)
lst.sort()
data=[]
train=[]
label=[]
totalnum = len(lst)
cnt=0
bearing1=[]
bearing1_label=[]
for filename in lst:
    cnt+=1
    rul = cnt/totalnum
    print(filename)
    image = cv2.imread('.//'+data_dir+'//'+filename)
    data.append(image)
    if(cnt>=num_timesteps):
        print(rul)
        label.append(rul)
        bearing1_label.append(rul)
        n = cnt-num_timesteps
        frame = data[n:]
        bearing1.append(frame)
        train.append(frame)

#Loading the Testing Data
print(data_dir2)
lst = os.listdir(data_dir2)
lst.sort()
data=[]
test=[]
cnt=0
for filename in lst:
    cnt+=1
    #print(filename)
    image = cv2.imread('.//'+data_dir2+'//'+filename)
    data.append(image)
    if(cnt>=num_timesteps):
        n = cnt-num_timesteps
        frame = data[n:]
        test.append(frame)

#Build Model
cnn = Sequential()
K1 = 10    # Conv1 layer feature map depth
K2 = 20    # Conv2 layer feature map depth
K3 = 40    # Conv3 layer feature map depth
K4 = 20    # Conv4 layer feature map depth
F1 = 500   # Full1 layer node size
F2 = 50    # Full2 layer node size

output = 1
#Add CNN layers
cnn.add(Conv2D(K2, kernel_size=(10, 10),strides=(2,2), activation='relu', input_shape=(128,128,3)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(K3, kernel_size=(5, 5),strides=(2,2), activation='relu'))
cnn.add(Conv2D(K4, kernel_size=(3, 3),strides=(1,1), activation='relu'))
cnn.add(Flatten())
#Add LSTM Layers
model = Sequential()
model.add(TimeDistributed(cnn, input_shape=(num_timesteps,128,128,3)))
model.add(LSTM(num_timesteps,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(num_timesteps,return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(num_timesteps))
model.add(Dropout(.2)) #added
model.add(Dense(output, activation='sigmoid'))

print(model.summary())

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
#Train the model
history = model.fit(train, label, batch_size=32, epochs=10) #Choose batch_size and epoch wisely

predict = model.predict(train)
plt.figure(1)
plt.plot(range(len(predict)),predict,color='b')
plt.plot(range(len(label)),label,color='r')
plt.savefig('train.png')

predict_test = model.predict(test)

#filling in the missing time for the numer of timesteps to fit the original time-series.
x=[]
for i in range(num_timesteps):
    x.append(0)
for i in range(len(predict)):
    x.append(i)
x = np.array(x)
y=[]
for i in range(num_timesteps):
    y.append(0)
for p in predict:
    y.append(p[0])
y = np.array(y)
print(x.shape)
print(y.shape)
d=[]
for i in range(len(x)):
    data=[]
    data.append(x[i])
    data.append(y[i])
    d.append(data)
d = np.array(d)
df = pd.DataFrame(d,columns=['Time','HI'])
#Saving the test result
df.to_csv('test_result.csv',sep=',')

plt.figure(2)
plt.plot(range(len(y)),y)
plt.savefig('test.png')
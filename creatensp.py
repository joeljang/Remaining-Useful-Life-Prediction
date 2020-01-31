import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
import csv
import tensorflow as tf
import cv2
import matplotlib.image as im
from sklearn import preprocessing
from sklearn.decomposition import PCA 
import sys

def lowPass(signal, cut, sample_length, sample_rate):
  ''' 
  low pass filter
  args:
    signal = input signal
    cut = filtering bandwidth Hz
    sample_length = total signal length (number of samples)
    sample_rate = sampling rate of the input signal
  return:
    filtered.real = filtered signal
  '''
  ratio = int(sample_length/sample_rate)
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.fft(signal)
    with tf.Session() as sess:
      tf.variables_initializer([signal]).run()
      result = sess.run(fft)
      for i in range(cut*ratio, len(result)-(cut*ratio)+1): #what is the intuition behind this?? 
        result[i] = 0
    ifft = tf.ifft(result)
    with tf.Session() as sess:
      filtered = sess.run(ifft)
  return filtered.real


def highPass(signal, cut, sample_length, sample_rate):
  ''' 
  high pass filter
  args:
    signal = input signal
    cut = filtering bandwidth Hz
    sample_length = total signal length (number of samples)
    sample_rate = sampling rate of the input signal
  return:
    filtered.real = filtered signal
  '''
  ratio = int(sample_length/sample_rate)
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.fft(signal)
    with tf.Session() as sess:
      tf.variables_initializer([signal]).run()
      result = sess.run(fft)
      for i in range(0, (cut*ratio)+1):
        result[i] = 0
      for i in range(len(result)-(cut*ratio), len(result)):
        result[i] = 0
    ifft = tf.ifft(result)
    with tf.Session() as sess:
      filtered = sess.run(ifft)
  return filtered.real


def bandPass(signal, low_cut, high_cut, sample_length, sample_rate):
  ''' 
  band pass filter
  args:
    signal = input signal
    low_cut = filtering bandwidth Hz (lower bound)
    high_cut = filtering bandwidth Hz (upper bound)
    sample_length = total signal length (number of samples)
    sample_rate = sampling rate of the input signal
  return:
    filtered.real = filtered signal
  '''
  ratio = int(sample_length/sample_rate)
  with tf.Graph().as_default():
    signal = tf.Variable(signal, dtype=tf.complex64)
    fft = tf.fft(signal)
    with tf.Session() as sess:
      tf.variables_initializer([signal]).run()
      result = sess.run(fft)
      for i in range(high_cut*ratio, len(result)-(high_cut*ratio)+1):
        result[i] = 0
      for i in range(0, (low_cut*ratio)+1):
        result[i] = 0
      for i in range(len(result)-(low_cut*ratio), len(result)):
        result[i] = 0
    ifft = tf.ifft(result)
    with tf.Session() as sess:
      filtered = sess.run(ifft)
  return filtered.real


def scatterPlot(x, y, gmin, gmax, size=1000):
  ''' 
  make scatter plot image
  args:
    x = channel 1 input list or 1D-array
    y = channel 2 input list or 1D-array
    gmin = minimum value to draw for pixel (1,1)
    gmax = maximum value to draw for pixel (size, size)
    size = image size = (size, size)
  return:
    splot = scatter plot image (numpy.ndarray)
  '''
  if type(x)==np.ndarray and type(y)==np.ndarray:
    pass
  elif type(x)==list and type(y)==list and len(x)*len(y)>0:
    x, y = np.array(x), np.array(y)
  else:
    print('ERROR: invalid type input x,y!\n       x,y must be list or numpy.ndarray')
    return None

  if gmin >= gmax:
    print('ERROR: invalid min, max bounds')
    return None

  splot = np.zeros([size, size])
  # normalize x and y
  x, y = x-gmin, y-gmin
  x, y = x/(gmax-gmin), y/(gmax-gmin)
  x, y = x*(size-1), y*(size-1)
  x, y = x.astype(int), y.astype(int)
  for i in range(len(x)):
      try: splot[x[i], y[i]] += 1
      except: continue
  splot /= splot.max() 
  splot *= 255 
  splot = np.uint8(splot)
  return splot

def toRGB(red, green, blue):
  if len(red) != len(green) or len(red) != len(blue):
    print('ERROR: different color channel image size')
    return None
  image = np.empty((len(red), len(red), 3), dtype=np.uint8)
  image[:,:,0] = red
  image[:,:,1] = green
  image[:,:,2] = blue
  return image


def processVibSignal(x, y, sample_length, sample_rate):
  band1 = [500,800]
  band2 = [800,900]
  band3 = [900,1200]
  #Choose the bandpass filters based on Fast-Fourier Transform analysis of time-frequency domain. Choose wisely. 

  x_red = bandPass(x, band1[0], band1[1], sample_length, sample_rate)
  y_red = bandPass(y, band1[0], band1[1], sample_length, sample_rate)
  x_green = bandPass(x, band2[0], band2[1], sample_length, sample_rate)
  y_green = bandPass(y, band2[0], band2[1], sample_length, sample_rate)
  x_blue = bandPass(x, band3[0], band3[1], sample_length, sample_rate)
  y_blue = bandPass(y, band3[0], band3[1], sample_length, sample_rate)
  red = scatterPlot(x_red, y_red, -1, 1, 128)
  green = scatterPlot(x_green, y_green, -1, 1, 128)
  blue = scatterPlot(x_blue, y_blue, -1, 1, 128)

  return toRGB(red, green, blue)

#Data loading and pre-processing
data_dir = sys.argv[1]
outputPath = sys.argv[2]

scaler = preprocessing.MinMaxScaler()

for filename in os.listdir(data_dir):
    print(filename)
    data = pd.read_csv(os.path.join(data_dir,filename), delimiter=',')
    data = np.array(data)
  
    x = data[:,4] #Get one column data. Customize depending on your data shape
    y = data[:,5] #Get one column data. Customize depending in your data shape
    length = x.shape[0]

    image = processVibSignal(x,y,length,length)
    im.imsave(outputPath+filename+'.png', image)
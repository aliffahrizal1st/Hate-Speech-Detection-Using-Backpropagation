# Hate Speech Detection Using Backpropagation in Indonesian Language

This project was build when i was working on internship in middle of 2020. Although it was a first commit but this project is already done at that time.

In this "Hate Speech Detection" model a.k.a Classification on Hate Speech, i was tasked to research some machine learning model and compare the results between them, and this is the one of it. There were two class that i'm using in this project one is 'Hate Speech' and other is 'Non-Hate Speech'.

It's all thanks to the researchers before me who already done working on some result that i can used for this research. Below are some of their repositories if you want to use them :
- [Stopwords](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_stop_words.txt) - it was combined stopwords made by [louisowen](https://github.com/louisowen6)
- [Normalization Dictionary](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/blob/master/new_kamusalay.csv) - this one really helps me to normalize the words in each data created by [okkyibrohim](https://github.com/okkyibrohim)
- [Dataset](https://github.com/ialfina/id-hatespeech-detection) - Last but not least the dataset that i used was made by [ialfina](https://github.com/ialfina)

And below are some explanations related to the methods in the BP model that I have created

## 1. Import Libraries
Download & import some libraries that are needed later
```
pip install sklearn
```
```
import pandas as pd
import math
import numpy as np
import random as ra
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files, drive
drive.mount('/content/gdrive')
```
## 2. One Hot Encoding on Dataset Target
This process is used to create a list containing classes from the training data. This may seem like it makes no difference but it will help us when coding the final result.
```
def oneHotData(label):
    for i in range(len(label)):
        if label[i]==1:
            label[i]=[1,0]
        elif label[i]==0:
            label[i]=[0,1]
    return label
```

## 3. Unique Word Extraction
In this process, only unique words from the entire document will be taken
```
def term(text):
  hasil=[]
  for i in text:
    for j in i.split():
      if j not in hasil:
        hasil.append(j)
  return hasil
```
## 4. Term Frequency (tf)
This method serves to count occurrences of all available features (words) in existing documents
```
def termfrequency(text,term):
  hasil = []
  text = [i.split() for i in text]
  for i in range(len(text)):
    hasil.append([])
    for j in range(len(term)):
      hasil[i].append(text[i].count(term[j]))
  return hasil
```
## 5. Parameter Initialization
After calculating the number of occurrences of all available features, the next step is to make some parameters that will be used in the Backpropagation model.
### 5.1 Creating network weights
In the weights created, 2 matrices will be generated with random values between -0.5 to 0.5 in the form of INxHN (number of input neurons x number of hidden neurons) and NHxON (number of hidden neurons x number of output neurons).
```
def weight(n_input,n_hidden,n_output):
  beta = 0.7 * (n_output**(1/n_input))
  #Hidden Layer
  hidden = []
  for i in range(n_hidden):
    hidden.append([])
    vector = 0
    for j in range(n_input):
      rand = ra.uniform(-0.5,0.5)
      hidden[i].append(rand)
      vector += rand**2
    hidden[i] = [(beta*y)/math.sqrt(vector) for y in hidden[i]]
  
  #Output Layer
  output = []
  for i in range(n_output):
    output.append([])
    for j in range(n_hidden):
      rand = ra.uniform(-0.5,0.5)
      output[i].append(rand)
  
  return hidden,output
```
### 5.2 Creating network bias
Almost the same as when making weights, in the biasing process, 2 matrices will be made in the form of 1xHN (number of layers x number of hidden neurons) and 1xON form, number of layers x number of output neurons) matrix with random values between -0.5 to 0.5. 
```
def bias(n_hidden,n_output):
  #Hidden Layer
  hidden = []
  for i in range(n_hidden):
    rand = ra.uniform(-0.5,0.5)
    hidden.append(rand)
  
  #Output Layer
  output = []
  for i in range(n_output):
    rand = ra.uniform(-0.5,0.5)
    output.append(rand)
  
  return hidden,output
```
## 6. Sigmoid
The two methods below are used to assist the mathematical process in performing sigmoid calculations and it derivative.
```
def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))
  
def derivative_sigmoid(x):
  return sigmoid(x)*(1.0-sigmoid(x))
```
## 7. Forward Propagation
After all the parameters have been obtained, we can make a calculation process, namely forward propagation.
```
def forward(termfreqTrain,w_hidden,w_output,b_hidden,b_output):
  #Hidden Layer
  in_hidden = np.add(np.dot(w_hidden,termfreqTrain.T),b_hidden)
  out_hidden = sigmoid(in_hidden)
  #Output Layer
  in_output = np.add(np.dot(w_output,out_hidden),b_output)
  out_output = sigmoid(in_output)
  return out_output,in_output,out_hidden,in_hidden
```
## 8. Backward Propagation
Furthermore, the backward propagation calculation process is carried out. There are several subprocesses in this process. First, checking the error value that exists between the results and the target. The second is to calculate the difference between the result and the actual target.
```
def backward(label,input,alpha,out_output,in_output,out_hidden,in_hidden):
  #Output Layer Error Checking 
  err_output = error_check_output(label,in_output,out_output)
  #Cek Delta Bobot Output Layer
  deltaWeightOutput = delta_bobot_output(alpha,err_output,out_hidden)
  #Cek Delta Bias Output Layer
  deltaBiasOutput = delta_bias_output(alpha,err_output)
  #Error Check Point dari Output Layer ke Hidden Layer
  checkpoint_error_hidden = error_hidden(err_output,w_output)
  #Hidden Layer Error Checking 
  err_hidden = error_check_hidden(checkpoint_error_hidden,in_hidden)
  #Cek Delta Bobot Hidden Layer
  deltaWeightHidden = delta_bobot_hidden(alpha,err_hidden,input)
  #Cek Delta Bias Hidden Layer
  deltaBiasHidden = delta_bias_hidden(alpha,err_hidden)
  return deltaWeightOutput,deltaBiasOutput,deltaWeightHidden,deltaBiasHidden
```
### 8.1 Error Checking
In the sub process, checking the error value between the results and the target is carried out from the existing layer, including the output layer, hidden layer and the output link between the hidden layer & output layer.
```
def error_check_output(or_label,in_output,out_output):
  return np.multiply(np.subtract(or_label,out_output),derivative_sigmoid(in_output))

def error_hidden(err_output,w_output):
  return np.dot(err_output,w_output)

def error_check_hidden(checkpoint_error_hidden,in_hidden):
  return np.multiply(checkpoint_error_hidden,derivative_sigmoid(in_hidden))
```
### 8.2 Delta Function
After the error value is obtained, we can calculate the value of the difference between the result and the actual target with a certain scale which will make it easier to update the parameters.
```
def delta_bobot_output(alpha,err_output,out_hidden):
  return (np.outer(err_output,out_hidden))*alpha

def delta_bias_output(alpha,err_output):
  return err_output*alpha

def delta_bobot_hidden(alpha,err_hidden,input):
  return (np.outer(err_hidden,input))*alpha

def delta_bias_hidden(alpha,err_hidden):
  return err_hidden*alpha
```

## 9.Update the Parameter
After that, the existing parameters are updated based on the difference between the results and the target. The updated parameters are bias & weight in both hidden layer and output layer. 
```
def update(w_hidden,w_output,b_hidden,b_output,deltaWeightOutput,deltaBiasOutput,deltaWeightHidden,deltaBiasHidden):
  #Update Output Layer Weight
  upWeightOutput = w_output+deltaWeightOutput
  #Update Output Layer Bias
  upBiasOutput = b_output+deltaBiasOutput
  #Update Hidden Layer Weight
  upWeightHidden = w_hidden+deltaWeightHidden
  #Update Hidden Layer Bias
  upBiasHidden = b_hidden+deltaBiasHidden
  return upWeightOutput,upBiasOutput,upWeightHidden,upBiasHidden
```
## 10. Training & Testing Process
Then we put all those methods in two main method that we used. The first one is used to training the model using the data, and the second is used to testing the model.
### 10.1 Traning
```
def training(epochs,data_train,train_label,w_hidden,w_output,b_hidden,b_output,alpha):
  w_output,b_output,w_hidden,b_hidden=w_output,b_output,w_hidden,b_hidden
  for epoch in range(epochs):
    for input,label in zip(data_train,train_label) :
      #Forward Propagation
      out_output,in_output,out_hidden,in_hidden = forward(input,w_hidden,w_output,b_hidden,b_output)
      if (out_output!=label).all:
        #Backward Propagation
        deltaWeightOutput,deltaBiasOutput,deltaWeightHidden,deltaBiasHidden = backward(label,input,alpha,out_output,in_output,out_hidden,in_hidden)
        #Update Weight
        upWeightOutput,upBiasOutput,upWeightHidden,upBiasHidden = update(w_hidden,w_output,b_hidden,b_output,deltaWeightOutput,deltaBiasOutput,deltaWeightHidden,deltaBiasHidden)
        w_output,b_output,w_hidden,b_hidden=upWeightOutput,upBiasOutput,upWeightHidden,upBiasHidden
    #print('epoch: ', epoch)
  return w_output,b_output,w_hidden,b_hidden
```
### 10.2 Testing 
```
def test(data_test,test_label,w_output,b_output,w_hidden,b_hidden):
  hasil = []
  for input in data_test:
    #Forward Propagation
    out_output,in_output,out_hidden,in_hidden = forward(input,w_hidden,w_output,b_hidden,b_output)
    out_output = oneHotFinal(out_output)
    hasil.append(out_output)
    #Cek Akurasi
  confusionMatrix(np.array(hasil),test_label)
```
## 11. One Hot Encode Label After Testing
After the test is done, the result we get is a probability value of both classes. And from the value of the two classes, the largest class value will determine the class of the existing test data.
```
def oneHotFinal(out_output):
    if (out_output[0]<out_output[1]):
        out_output = [0,1]
    elif (out_output[0]>out_output[1]):
        out_output = [1,0]
    return out_output
```
## 12. Evaluation
In the final process, we evaluate all the final labels that our model has created. This process uses the Confusion Matrix as one of the simplest evaluation methods.
```
def confusionMatrix(hasil,test_label):
  f, ax = plt.subplots(figsize=(8,5))
  sns.heatmap(confusion_matrix(hasil.argmax(axis=1), test_label.argmax(axis=1)), annot=True, fmt=".0f", ax=ax)
  plt.xlabel("y_head")
  plt.ylabel("y_true")
  plt.show()
  print (classification_report(hasil, test_label))
```

#!/usr/bin/env python
# coding: utf-8

# In[1]:

#docker run --gpus all -it --rm -v ${pwd}:/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter

get_ipython().system('pip install transformers==4.1.1 pandas==1.1.5 scikit-learn==0.24.0')


# In[2]:


import transformers
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[3]:


df=pd.read_csv('imdb.csv', header=None)
df.columns=['review','label']

df['label']=df['label'].map({'Negative':0.0,'Positive':1.0})


# In[4]:


df.describe()


# In[5]:


Xtrain, ytrain = df['review'], df['label']
Xtrain,Xval,ytrain,yval=train_test_split(Xtrain, ytrain, test_size=0.2,random_state=10)


# In[6]:


tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')


# In[7]:


#pass our texts to the tokenizer. 
Xtrain_enc = tokenizer(Xtrain.tolist(), 
                         truncation=True, padding=True, 
                         add_special_tokens=True, return_tensors='np') #return numpy object
Xval_enc = tokenizer(Xval.tolist(), 
                         truncation=True, padding=True, 
                         add_special_tokens=True, return_tensors='np') #return numpy object


# In[8]:


#preparing our datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Xtrain_enc),
    ytrain
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Xval_enc),
    yval
))


# In[9]:


yval


# In[10]:


def bert_model(train_dataset,val_dataset,transformer,max_len,epochs):
    print("----Building the model----")
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,),dtype=tf.int32,name = 'attention_mask') #attention mask
    sequence_output = transformer(input_ids,attention_mask)[0]
    cls_token = sequence_output[:, 0, :]
    x = Dense(512, activation='relu')(cls_token)
    x = Dropout(0.1)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_ids,attention_mask], outputs=y)
    model.summary()
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    r = model.fit(train_dataset.batch(32),batch_size = 32,
                  validation_data = val_dataset.batch(32),epochs = epochs)
                  #callbacks = callbacks
    print("Train score:", model.evaluate(train_dataset.batch(32)))
    print("Validation score:", model.evaluate(val_dataset.batch(32)))
    n_epochs = len(r.history['loss'])
    
    return r,model,n_epochs 


# In[11]:


transformer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')


# In[12]:


epochs = 2
max_len = 512
r,model,n_epochs = bert_model(train_dataset,val_dataset,transformer,max_len,epochs)


# In[16]:


model.save("./sentiment")


# 

# In[ ]:





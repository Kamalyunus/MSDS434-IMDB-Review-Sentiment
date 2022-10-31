#docker build -t tensorflow . -f .\train_model\Dockerfile
#docker run --gpus all -it tensorflow -v ${pwd}:/root
#docker run --gpus all -it --rm -v ${pwd}:/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter

import transformers
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def prep_data(path):
    print('reading csv file...')
    df=pd.read_csv(path, header=None)
    df.columns=['review','label']
    df['label']=df['label'].map({'Negative':0.0,'Positive':1.0})
    Xtrain, ytrain = df['review'], df['label']
    print('splitting data...')
    Xtrain,Xval,ytrain,yval=train_test_split(Xtrain, ytrain, test_size=0.2,random_state=10)
    print('tokenizing data...')
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    Xtrain_enc = tokenizer(Xtrain.tolist(), 
                         truncation=True, padding=True, 
                         add_special_tokens=True, return_tensors='np') #return numpy object
    Xval_enc = tokenizer(Xval.tolist(), 
                         truncation=True, padding=True, 
                         add_special_tokens=True, return_tensors='np') #return numpy object
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(Xtrain_enc),
        ytrain))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(Xval_enc),
        yval))

    return train_dataset, val_dataset

def bert_model(train_dataset,val_dataset,epochs):
    print("----Building the model----")
    max_len=512
    transformer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,),dtype=tf.int32,name = 'attention_mask') #attention mask
    sequence_output = transformer(input_ids,attention_mask)[0]
    cls_token = sequence_output[:, 0, :]
    x = Dense(512, activation='relu')(cls_token)
    x = Dropout(0.1)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_ids,attention_mask], outputs=y)
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print('model fitting...')
    model.fit(train_dataset.batch(32),batch_size = 32,
                validation_data = val_dataset.batch(32),epochs = epochs)
    print("Train score:", model.evaluate(train_dataset.batch(32)))
    print("Validation score:", model.evaluate(val_dataset.batch(32)))
    
    return model

if __name__=='__main__':
    print('main')
    train_dataset, val_dataset = prep_data('imdb.csv')
    model = bert_model(train_dataset,val_dataset,epochs=2)
    print('Saving model...')
    model.save("./sentiment")



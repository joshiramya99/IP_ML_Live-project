#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.utils
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from numpy import array
from numpy import asarray
from numpy import zeros



# In[2]:


#Fetching event type dataset
df1_type=pd.read_csv("event_type.csv",encoding='latin1')
df2_type=df1_type.iloc[ : ,[1]]
df2_type=df2_type.replace(np.nan,'')
l1=list()
l1=df2_type.values.tolist()
df2_type['total']=pd.Series(l1).values
df2_type['total']=df2_type['total'].apply(set)



# In[3]:


#Fetching event domain dataset
df1_domain=pd.read_csv("event_domainfinal.csv",encoding='latin1')
df2_domain=df1_domain.iloc[ : ,[1]]
df2_doamin=df2_domain.replace(np.nan,'')
l2=list()
l2=df2_domain.values.tolist()
df2_domain['total']=pd.Series(l2).values
df2_domain['total']=df2_domain['total'].apply(set)


# In[7]:


# one hot encoding of labels 
mlb = MultiLabelBinarizer()
l_type=mlb.fit_transform(df2_type['total'])
df3_type=pd.DataFrame(l_type)
df3_type.columns=list(mlb.classes_)
#df3=df3.drop(columns=[''])
#print(mlb.classes_)
data=pd.Series(df1_type['Event'])
df3_type.insert(0,'Event',data)
df3_type = sklearn.utils.shuffle(df3_type)
df3_type = df3_type.reset_index(drop=True)
#print(df3_type)


l_domain=mlb.fit_transform(df2_domain['total'])
df3_domain=pd.DataFrame(l_domain)
df3_domain.columns=list(mlb.classes_)
#df3=df3.drop(columns=[''])
#print(mlb.classes_)
data=pd.Series(df1_domain['Event'])
df3_domain.insert(0,'Event',data)
df3_domain = sklearn.utils.shuffle(df3_domain)
df3_domain = df3_domain.reset_index(drop=True)
#print(df3_domain.head())



# In[8]:



def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence =str(re.sub('[^0-9a-zA-Z+]', ' ',str(sen)))

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
   
    sentence=str.lower(sentence)
    
    text_tokens = word_tokenize(sentence)

    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)

    return filtered_sentence


# In[9]:


x_type = []
x_domain=[]
sentences1 = list(df3_type["Event"])
sentences2 = list(df3_domain["Event"])
for sen in sentences1:
    x_type.append(preprocess_text(sen))
for sen in sentences2:
    x_domain.append(preprocess_text(sen))


    


# In[10]:


#Converting text into tokens
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(x_type)
X_data_type = tokenizer1.texts_to_sequences(x_type)
vocab_size1 = len(tokenizer1.word_index) + 1

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(x_domain)
X_data_domain = tokenizer2.texts_to_sequences(x_domain)
vocab_size2 = len(tokenizer2.word_index) + 1

maxlen = 200
X_data_type = pad_sequences(X_data_type, padding='post', maxlen=maxlen)
X_data_domain = pad_sequences(X_data_domain, padding='post', maxlen=maxlen)
#print(X_data_type.shape)
#print(X_data_domain.shape)


# In[11]:




event_type=df3_type[['Certifications','Competitions','Courses','Expos','Fests','Hackathons','Internships','Jobs','Talks','Training','Webinars','Workshops']]
event_domain=df3_domain[['Artificial Intelligence' ,'Blockchain', 'C' ,'C++' ,'Cloud Computing',
 'Coding', 'Data Science', 'Development Process', 'Finance' ,'Hardware',
 'Higher Education' ,'IoT', 'Java' ,'Javascript', 'Machine Learning',
 'Management' ,'Mobile Applications', 'Networking' ,'Other','Python',
 'Security' ,'Software Architecture', 'Web Development']]
y_type = event_type.values
y_domain = event_domain.values
#print(y_type.shape)
#print(y_domain.shape)


# In[12]:


embeddings_dictionary = dict()

glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()


# In[13]:


#Mapping words into embedding vectors
embedding_matrix1 = zeros((vocab_size1, 100))
for word, index in tokenizer1.word_index.items():
    embedding_vector1 = embeddings_dictionary.get(word)
    if embedding_vector1 is not None:
        embedding_matrix1[index] = embedding_vector1


# In[14]:


embedding_matrix2 = zeros((vocab_size2, 100))
for word, index in tokenizer2.word_index.items():
    embedding_vector2 = embeddings_dictionary.get(word)
    if embedding_vector2 is not None:
        embedding_matrix2[index] = embedding_vector2


# In[15]:


#Model of classifying event type
input1 = tf.keras.layers.Input(shape=(maxlen,))
 
x1 = tf.keras.layers.Embedding(vocab_size1,100, weights=[embedding_matrix1], trainable=False)(input1)

x1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x1)
 
x1= tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x1)
 
avg_pool1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
max_pool1 = tf.keras.layers.GlobalMaxPooling1D()(x1)
 
x1 = tf.keras.layers.concatenate([avg_pool1, max_pool1])
 
preds1 = tf.keras.layers.Dense(12, activation="sigmoid")(x1)
 
model1 = tf.keras.Model(input1, preds1)
 
#model1.summary()


# In[16]:


model1.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
history1 = model1.fit(X_data_type, y_type,epochs=5, validation_split=0.2,class_weight='balanced',shuffle=False)


# In[17]:


#Model for classifying event domain
input2 = tf.keras.layers.Input(shape=(maxlen,))
 
x2 = tf.keras.layers.Embedding(vocab_size2,100, weights=[embedding_matrix2], trainable=False)(input2)

x2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x2)
 
x2 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x2)
 
avg_pool2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
max_pool2 = tf.keras.layers.GlobalMaxPooling1D()(x2)
 
x2 = tf.keras.layers.concatenate([avg_pool2, max_pool2])
 
preds2 = tf.keras.layers.Dense(23, activation="sigmoid")(x2)
 
model2 = tf.keras.Model(input2, preds2)
 
#model1.summary()


# In[18]:


model2.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
history2 = model2.fit(X_data_domain, y_domain,epochs=5, validation_split=0.2,class_weight='balanced',shuffle=False)


# In[294]:


tokenizer = Tokenizer()
labels_domain=['Artificial Intelligence' ,'Blockchain', 'C' ,'C++' ,'Cloud Computing',
 'Coding', 'Data Science', 'Development Process', 'Finance' ,'Hardware',
 'Higher Education' ,'IoT', 'Java' ,'Javascript', 'Machine Learning',
 'Management' ,'Mobile Applications', 'Networking','Other','Python',
 'Security' ,'Software Architecture', 'Web Development']
labels_type=['Certifications','Competitions','Courses','Expos','Fests','Hackathons','Internships','Jobs','Talks','Trainings','Webinars','Workshops']

#Function for predicting type and domain
def classify_event_type(x) :
    event_set=pd.read_csv(x,encoding='latin1')
    sentences = list(event_set["Event"])
    
    xvar=[]
    type_list=[]
    domain_list=[]
    for sen in sentences:
        xvar.append(preprocess_text(sen))
    
    tokenizer.fit_on_texts(xvar)

    sen_data = tokenizer.texts_to_sequences(xvar)

    maxlen = 200

    sen_data = pad_sequences(sen_data, padding='post', maxlen=maxlen)
    #print(sen_data.shape)
    types=model1.predict(sen_data)
    domains=model2.predict(sen_data)
    list1=np.argmax(types,axis=1)
    list2=np.argmax(domains,axis=1)
    for var in list1:
        type_list.append(labels_type[var])
    for var in list2:
        domain_list.append(labels_domain[var])
    #print(type_list)
    #print(domain_list)
    emp_list=match(type_list,domain_list)
    c=[]
    for i in range(len(emp_list)):
        c.append(", ".join(emp_list[i]))
                   
        event_set['Name']=pd.Series(c)
    #print(event_set)
    event_set.to_excel("output.xlsx")
    


# In[295]:


#Matching predicted type and domain with employee database
def match(type_list,domain_list) :
    emp_list=[]
    emp=pd.read_csv("CCMLEmployeeData.csv",encoding='latin1')
   
    for x in range(len(type_list)) :
        z=[];
        for index, row in emp.iterrows():
            if(domain_list[x]!='Other'):
                if((row['Event1']==type_list[x] or row['Event2']==type_list[x])and row['Domain']==domain_list[x]):
                    z.append(row['Name'])
            else :
                if(row['Event1']==type_list[x] or row['Event2']==type_list[x]):
                    z.append(row['Name'])


        emp_list.append(z)

    return(emp_list)


# In[297]:


classify_event_type("test.csv")


# In[ ]:





# In[ ]:





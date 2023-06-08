
#%%
#1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import os,datetime
# %%
#2. Data loading
file_path = r"C:\Users\Admin\Documents\YP03-DALILAH\DEEP LEARNING\HANDS-ON\Assessement2\ecommerceDataset.csv"
data = pd.read_csv(file_path)
# %%
#3. Data inspection
#4. Data inspection
print("Shape of data = ", data.shape)
print("missing data =",data.isna().sum())
print(data.describe().transpose())
print(data.info())
print(data.head(1))
print("duplicated=",data.duplicated().sum())
#%%
#5. Data cleaning
#duplicate
data = data.drop_duplicates()
print("duplicated=",data.duplicated().sum())
#%%
#missing data
data = data.dropna(axis=0)
print("missing data =",data.isna().sum())
print("Shape of data = ", data.shape)
# %%
#6. Split into features and labels
features = data['text'].values
labels = data['label'].values
# %%
#5. Convert label into integers using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_processed = label_encoder.fit_transform(labels)

# %%
#6. Data preprocessing
#(A) Remove unwanted strings from the data
from review_handler import remove_unwanted_strings

feature_removed = remove_unwanted_strings(features)
# %%
#7. Define some hyperparameters
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8
# %%
#8. Perform train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(feature_removed,label_processed,train_size=training_portion,random_state=12345)
# %%
#9. Perform tokenization
from tensorflow import keras

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,split=" ",oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
#%%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
# %%
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
# %%
#10. Perform padding and truncating
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens,maxlen=(max_length))
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens,maxlen=(max_length))
# %%
#11. Model development
#(A) Create the sequential model
model = keras.Sequential()
#(B) Create the input layer, in this case, it can be the embedding layer
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Create the bidirectional LSTM layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)))
#(C) Classification layers
model.add(keras.layers.Dense(embedding_dim,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(y_train)),activation='softmax'))

model.summary()
# %%
#12. Model compilation
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#%%
#10.Create a TensorBoard callback object for the usage of TensorBoard
base_log_path = r"tensorbaord_logs\ecomerce"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#13. Model training
history = model.fit(X_train_padded,y_train,validation_data=(X_test_padded,y_test),epochs=50,batch_size=64,callbacks=[tb])
# %%
#14. Model evaluation
print(history.history.keys())
# %%
#Plot accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train accuracy","Test accuracy"])
plt.show()
# %%
#15. Model deployment
test_string = ['Introducing a new hair attachment for the Dyson Supersonic hair dryer, equipping stylers with a solution to smooth stray strands without extreme heat.']

# %%
test_string_removed = remove_unwanted_strings(test_string)
#%%
test_string_tokens = tokenizer.texts_to_sequences(test_string_removed)
#%%
test_string_padded = keras.preprocessing.sequence.pad_sequences(test_string_tokens,maxlen=(max_length))

# %%
y_pred = np.argmax(model.predict(test_string_padded),axis=1)

# %%
label_map = ['Household','Books','Clothing & Accessories','Electronic']
predicted_sentiment = [label_map[i] for i in y_pred]

# %%
#16. Save model and tokenizer
import os

PATH = os.getcwd()
print(PATH)
# %%
#Model save path
model_save_path = os.path.join(PATH,"ecomerce_model")
keras.models.save_model(model,model_save_path)
#%%
#Check if the model can be loaded
model_loaded = keras.models.load_model(model_save_path)
#%%
#tokenizer save path
import pickle

tokenizer_save_path = os.path.join(PATH,"tokenizer.pkl")
with open(tokenizer_save_path,'wb') as f:
    pickle.dump(tokenizer,f)

#%%
#Check if the tokenizer object can be loaded
with open(tokenizer_save_path,'rb') as f:
    tokenizer_loaded = pickle.load(f)

# %%

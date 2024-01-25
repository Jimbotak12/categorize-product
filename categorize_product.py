# %%
# setup_import package
import numpy as np
import pandas as pd
import seaborn as sns 
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
from keras.models import Sequential
import os, pickle, datetime, json, sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Bidirectional, Dense
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# %%
# Load data
df = pd.read_csv('ecommerceDataset.csv')
# %%
# Set the columns name
df.columns = ['category','description']
# %%
# Data inspection
print("Shape of data:",df.shape)
print("\n--------------\n")
print("Data info:\n",df.info())
print("\n--------------\n")
print("Data describe:\n",df.describe().transpose())
print("\n--------------\n")
print("Example data:\n",df.head())
# %%
# Count each category
categories_list = df['category'].unique()
print(df['category'].value_counts())
# %%
# Data cleaning
print("NaN Data:\n",df.isna().sum())
print("")
print("Duplicate Data: ",df.duplicated().sum())
# %%
# Remove the NaN
df = df.dropna()
# %%
# Data preprocessing
# Split the data into features ana labels
features = df['description'].values
labels = df['category'].values
# %%
# Convert the categorical label into interger - label encoding
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(labels)
# %%
# Perform train test split
seed = 42
X_train,X_test,y_train,y_test = train_test_split(features, label_encoded, train_size=0.8, random_state=seed)
# %%
# Process the input texts
# Tokenizer
# Define parameters for the following process
vocab_size =5000 # 5000 will converted into number
oov_token = '<OOV>'
max_length = 200 # use when doing the padding
embedding_dim = 64 # use for embedding process

# Define the Tokenizer object
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    split=" ",
    oov_token=oov_token
)
tokenizer.fit_on_texts(X_train)
# %%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))
# %%
# Transform texts into token
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)
# %%
# Perform padding
X_train_padden = keras.utils.pad_sequences(
    X_train_tokens,
    maxlen=max_length,
    padding="post",
    truncating="post" # cut from behind
)

X_test_padden = keras.utils.pad_sequences(
    X_test_tokens,
    maxlen=max_length,
    padding="post",
    truncating="post" # cut from behind
)
# %%
# Model development
# Create a sequantial model, then start with embedding layer
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(48)))
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(8,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(labels)), activation='softmax'))

print(model.summary())
# %%
# Compile the model
model.compile(
    optimizer='adam', 
    loss=keras.losses.sparse_categorical_crossentropy, 
    metrics=["accuracy"]
    )
# %%
# Model training
max_epoch = 20
# Prepare the callback object for model.fit()
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
    )
# Perfoemance graph at tensorboard
path = os.getcwd()
logpath = os.path.join(path,"tensorboard_log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(logpath)
# %%
hist = model.fit(
    X_train_padden,
    y_train,
    validation_data=(X_test_padden, y_test),
    epochs=max_epoch,
    callbacks=[early_stopping, tb]
)
# %%
# Plot graph to display 
# loss graph
fig = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()
# Accuracy graph
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='blue', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='lower right')
plt.show()
# %%
y_pred = model.predict(X_test_padden)
y_pred_index = np.argmax(y_pred,axis=1)

print(classification_report(y_test, y_pred_index))
# %%
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_index)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['0','1','2','3'], yticklabels=['0','1','2','3'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# %%
#13. Save important comonents so that we can deploy the NLP model in other app
# Get directory
os.chdir(r"C:\Users\muhdh\OneDrive\Desktop\Hands_on\capston\Assesmen 3")
PATH = os.getcwd()
# %%
# Tokenizer
tokenizer_save_path = os.path.join(PATH,"save_model","tokenizer.pkl")
with open(tokenizer_save_path,"wb") as f:
    pickle.dump(tokenizer,f)
# %%
# Label encoder
label_encoder_save_path = os.path.join(PATH,"save_model","label_encoder.pkl")
with open(label_encoder_save_path,"wb") as f:
    pickle.dump(label_encoder,f)
# %%
# Save the keras model
model_save_path = os.path.join(PATH,"save_model","nlp_categorize_product.h5")
keras.models.save_model(model,model_save_path)
# %%

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import json
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import unidecode

# Cargar datos de intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Importar tokenización, stopwords y lematización de nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configurar lematizador y stop words en español
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Función para preprocesar el texto (sin spaCy y TextBlob)
def preprocess_text(text):
    # Quitar las tildes y convertir a minúsculas
    text = unidecode.unidecode(text.lower())
    
    # Tokenizar el texto
    tokens = word_tokenize(text)

    # Filtrar stop words y lematizar las palabras
    processed_words = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words and word.isalpha()  # Eliminar signos de puntuación y números
    ]
    
    return ' '.join(processed_words)

# Preprocesar los patrones y las etiquetas
patterns = []
tags = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        processed_pattern = preprocess_text(pattern)
        patterns.append(processed_pattern)
        tags.append(intent["tag"])

# Luego puedes continuar con el proceso de vectorización y entrenamiento del modelo


# Paso 6: Convertir los patrones en características utilizando CountVectorizer
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(patterns).toarray()

# Convertir las etiquetas en números utilizando LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 7: Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(set(tags)), activation="softmax"))

# Compilar el modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=200, batch_size=4, verbose=1, validation_data=(X_test, y_test))

model.save("chatbot_model.h5")
np.save("classes.npy", label_encoder.classes_)
np.save("vectorizer.npy", vectorizer)
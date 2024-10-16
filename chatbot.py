import numpy as np
import random
import json
import os
from tensorflow.keras.models import load_model

# Cargar el archivo intents.json usando rutas seguras
base_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(base_dir, "intents.json")

with open(intents_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Cargar el modelo entrenado
model_path = os.path.join(base_dir, "chatbot_model.h5")
model = load_model(model_path)

# Cargar el vectorizer y label_encoder usando npy
vectorizer_path = os.path.join(base_dir, "vectorizer.npy")
vectorizer = np.load(vectorizer_path, allow_pickle=True).item()

# Cargar las clases desde el archivo classes.npy
label_encoder_path = os.path.join(base_dir, "classes.npy")
classes = np.load(label_encoder_path, allow_pickle=True)

# Variables globales
preferencia = None  # Variable global para almacenar la preferencia del usuario
ultima_opcion = None  # Variable para recordar el último intent de cena

# Función para preprocesar el texto
def preprocess_text(text):
    return text.lower()

# Función principal del chatbot
def chatbot_response1(user_input):
    global preferencia, ultima_opcion

    # Inicializar intent_tag
    intent_tag = None

    # Convertir la entrada del usuario en un formato que el modelo pueda usar
    input_data = vectorizer.transform([preprocess_text(user_input)]).toarray()
    prediction = model.predict(input_data)

    # Obtener la probabilidad más alta y su índice
    intent_index = np.argmax(prediction)
    max_prob = prediction[0][intent_index]  # Obtener la probabilidad más alta

    # Definir un umbral de confianza
    confidence_threshold = 0.4  # 40% de confianza

# Imprimir las predicciones para depuración
    print(f"Predicciones: {prediction}")
    print(f"Probabilidad máxima: {max_prob}, Intento predicho: {classes[intent_index]}")

    # Si la probabilidad es mayor que el umbral, devolver el intent
    if max_prob >= confidence_threshold:
        intent_tag = classes[intent_index]

        # Si el intent es "cena" y aún no tenemos una preferencia, preguntar sobre la preferencia
        if intent_tag == "cena" and preferencia is None:
            preferencia = "preguntada"
            return "¿Prefieres una opción con carne o vegetariana?"

        # Si el intent es "otra_opcion", ofrecer otra opción basada en la preferencia ya seleccionada
        if intent_tag == "otra_opcion" and preferencia is not None:
            intent_tag = ultima_opcion  # Mantener la última preferencia de cena (carne o vegetariana)

        # Si el intent es de despedida o agradecimiento, restablecer la preferencia
        if intent_tag in ["despedida", "agradecimiento"]:
            preferencia = None  # Reiniciar la preferencia
            for intent in intents["intents"]:
                if intent["tag"] == intent_tag:
                    response = random.choice(intent["responses"])
                    return response

    # Si el chatbot ya preguntó sobre las preferencias
    if preferencia == "preguntada":
        if "carne" in user_input.lower():
            preferencia = "carne"
            intent_tag = "cena_carne"
        elif "vegetariana" in user_input.lower():
            preferencia = "vegetariana"
            intent_tag = "cena_vegetariana"
        else:
            return "Lo siento, no entendí tu preferencia. ¿Carne o vegetariana?"

    # Si intent_tag no se asignó ningún valor (es decir, no se encontró una intención con suficiente confianza)
    if intent_tag is None:
        return "Lo siento, no entendí tu mensaje."

    # Buscar una respuesta aleatoria para el intent predicho
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            response = random.choice(intent["responses"])
            break

    # Guardar el intent actual como la última opción seleccionada (carne o vegetariana)
    if intent_tag in ["cena_carne", "cena_vegetariana"]:
        ultima_opcion = intent_tag  # Mantener la preferencia para "otra opción"

    return response
from flask import Flask, render_template, request, jsonify  # Importa Flask y funciones para manejar las peticiones y respuestas JSON
from chatbot import chatbot_response1  # Importa la función chatbot del archivo chatbot.py

# Inicializa la aplicación Flask
app = Flask(__name__)

# Ruta principal que renderiza el archivo index.html
@app.route('/')
def index():
    return render_template('index.html')

# Ruta que maneja las interacciones con el chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Recibe el mensaje del usuario desde el frontend
    data = request.get_json()
    user_message = data.get('message')

    # Genera la respuesta del chatbot utilizando la función importada
    bot_response = chatbot_response1(user_message)

    # Devuelve la respuesta del chatbot en formato JSON al frontend
    return jsonify({'response': bot_response})

# Ejecuta la aplicación en modo de depuración
if __name__ == '__main__':
    app.run(debug=True)
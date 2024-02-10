from flask import Flask, request, jsonify
import threading
import json
import os
from kafka import KafkaConsumer, TopicPartition
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
source_topic_name = os.getenv('SOURCE_TOPIC_NAME')
bootstrap_servers = [os.getenv('BROKER')]
group_id = os.getenv('GROUP_ID')

global interpreter, tokenizer, max_sequence_len
interpreter = None
tokenizer = None
max_sequence_len = None

def load_model_and_tokenizer(payload):
    global interpreter, tokenizer, max_sequence_len
    model_base64 = payload['model']
    tflite_model = base64.b64decode(model_base64)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    tokenizer_json = payload['tokenizer']
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    max_sequence_len = input_details[0]['shape'][1]
    print("Model and tokenizer updated.")

def update_model_from_kafka():
    consumer = KafkaConsumer(
        source_topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id=group_id,
        value_deserializer=lambda x: x,
        fetch_max_bytes=314572800,
        max_partition_fetch_bytes=314572800
    )
    consumer.subscribe([source_topic_name])

    while True:
        message_pack = consumer.poll(timeout_ms=10000)
        for topic_partition, messages in message_pack.items():
            for message in messages:
                try:
                    latest_payload = json.loads(message.value.decode('utf-8'))
                    if 'model' in latest_payload and 'tokenizer' in latest_payload:
                        load_model_and_tokenizer(latest_payload)
                except json.JSONDecodeError:
                    print("Failed to decode JSON from message. Skipping.")
                except Exception as e:
                    print(f"Error updating model and tokenizer: {e}")

@app.route('/generate-text', methods=['POST'])
def generate_text_api():
    if not interpreter or not tokenizer:
        return jsonify({"error": "Model or tokenizer not loaded"}), 500

    data = request.json
    seed_text = data.get('seed_text', '')
    num_generate = data.get('num_generate', 20)

    generated_text = generate_text(seed_text, num_generate)
    return jsonify({"generated_text": generated_text})

def generate_text(seed_text, num_generate):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    
    generated_text = []
    for _ in range(num_generate):
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], token_list.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        predicted = np.argmax(output_data, axis=-1)
        
        if predicted.size == 0:
            break
        next_token = predicted[0]
        generated_text.append(next_token)
        token_list = np.append(token_list[:,1:], [[next_token]], axis=-1)

    generated_text_words = [tokenizer.index_word[token] if token in tokenizer.index_word else "" for token in generated_text]
    return ' '.join(generated_text_words)

if __name__ == '__main__':
    threading.Thread(target=update_model_from_kafka, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5050)

import threading
import json
import os
import base64
from kafka import KafkaConsumer
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv

class TextGeneratorService:
    def __init__(self):
        load_dotenv()
        self.source_topic_name = os.getenv('SOURCE_TOPIC_NAME')
        self.bootstrap_servers = [os.getenv('BROKER')]
        self.group_id = os.getenv('GROUP_ID')

        self.interpreter = None
        self.tokenizer = None
        self.max_sequence_len = None

        thread = threading.Thread(target=self.update_model_from_kafka, daemon=True)
        thread.start()

    def load_model_and_tokenizer(self, payload):
        model_base64 = payload['model']
        tflite_model = base64.b64decode(model_base64)
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

        tokenizer_json = payload['tokenizer']
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

        input_details = self.interpreter.get_input_details()
        self.max_sequence_len = input_details[0]['shape'][1]
        print("Model and tokenizer updated.")

    def update_model_from_kafka(self):
        consumer = KafkaConsumer(
            self.source_topic_name,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            group_id=self.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            fetch_max_bytes=314572800,
            max_partition_fetch_bytes=314572800
        )

        for message in consumer:
            try:
                if 'model' in message.value and 'tokenizer' in message.value:
                    self.load_model_and_tokenizer(message.value)
            except json.JSONDecodeError:
                print("Failed to decode JSON from message. Skipping.")
            except Exception as e:
                print(f"Error updating model and tokenizer: {e}")

    def generate_text(self, seed_text, num_generate):
        if not self.interpreter or not self.tokenizer:
            return "Model and tokenizer not loaded yet."

        token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=self.max_sequence_len, padding='pre')
        
        generated_text = []
        for _ in range(num_generate):
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], token_list.astype(np.float32))
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
            predicted = np.argmax(output_data, axis=-1)
            
            if predicted.size == 0:
                break
            next_token = predicted[0]
            generated_text.append(next_token)
            token_list = np.append(token_list[:,1:], [[next_token]], axis=-1)

        generated_text_words = [self.tokenizer.index_word[token] if token in self.tokenizer.index_word else "" for token in generated_text]
        return ' '.join(generated_text_words)

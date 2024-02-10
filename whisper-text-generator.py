from kafka import KafkaConsumer, TopicPartition
import json
import os
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

load_dotenv()

source_topic_name = os.getenv('SOURCE_TOPIC_NAME')
bootstrap_servers = [os.getenv('BROKER')]
group_id = os.getenv('GROUP_ID')

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

consumer.poll(timeout_ms=1000)

# Seek to the last message for each partition
for partition in consumer.partitions_for_topic(source_topic_name):
    tp = TopicPartition(source_topic_name, partition)
    consumer.seek_to_end(tp)
    last_offset = consumer.position(tp) - 1
    if last_offset >= 0:
        consumer.seek(tp, last_offset)
    else:
        print("No models in the topic? :((((")

latest_payload = None
for message in consumer:
    latest_payload = json.loads(message.value.decode('utf-8'))
    break

if latest_payload:
    model_base64 = latest_payload['model']
    tflite_model = base64.b64decode(model_base64)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    tokenizer_json = latest_payload['tokenizer']
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    max_sequence_len = input_details[0]['shape'][1]
else:
    print("No model or tokenizer found.")
    exit()

def generate_text(seed_text, num_generate):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    
    generated_text = []

    for _ in range(num_generate):
        interpreter.set_tensor(input_details[0]['index'], token_list.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output_data, axis=-1)
        
        if predicted.size == 0:
            break
        next_token = predicted[0]
        generated_text.append(next_token)
        token_list = np.append(token_list[:,1:], [[next_token]], axis=-1)

    # Convert token IDs back to words
    generated_text_words = [tokenizer.index_word[token] if token in tokenizer.index_word else "" for token in generated_text]
    return ' '.join(generated_text_words)

# Generate and print text
generated_text = generate_text(seed_text="hot dog grex minecraft hot dog", num_generate=20)
print("Generated Text:\n", generated_text)

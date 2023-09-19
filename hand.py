import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the characters to be used in the generated text
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
num_characters = len(characters)

# Generate synthetic data for training (random sequences of characters)
num_samples = 10000
max_sequence_length = 50

data = []
for _ in range(num_samples):
    sequence_length = np.random.randint(10, max_sequence_length)
    random_sequence = ''.join(random.choice(characters) for _ in range(sequence_length))
    data.append(random_sequence)

# Create a mapping from characters to integers and vice versa
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for idx, char in enumerate(characters)}

# Preprocess the data and create input-output pairs
sequence_length = 40
X_data = []
y_data = []

for sequence in data:
    if len(sequence) < sequence_length + 1:
        continue
    for i in range(len(sequence) - sequence_length):
        input_seq = sequence[i:i + sequence_length]
        output_seq = sequence[i + sequence_length]
        X_data.append([char_to_idx[char] for char in input_seq])
        y_data.append(char_to_idx[output_seq])

X_data = np.array(X_data)
y_data = np.array(y_data)

# One-hot encode the input sequences
X_one_hot = np.zeros((X_data.shape[0], sequence_length, num_characters), dtype=bool)
for i, sequence in enumerate(X_data):
    for t, char_idx in enumerate(sequence):
        X_one_hot[i, t, char_idx] = 1

# Build the RNN model
model = keras.Sequential([
    layers.LSTM(128, input_shape=(sequence_length, num_characters)),
    layers.Dense(num_characters, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X_one_hot, y_data, batch_size=64, epochs=20)

# Generate text using the trained model
def generate_text(seed_text, length=100):
    generated_text = seed_text
    for _ in range(length):
        input_sequence = [char_to_idx[char] for char in generated_text[-sequence_length:]]
        input_sequence = np.array([input_sequence], dtype=np.int32)
        predicted_prob = model.predict(input_sequence)[0]
        next_char_idx = np.random.choice(num_characters, p=predicted_prob)
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char

    return generated_text

# Generate text starting with a seed
seed_text = "This is a sample generated text: "
seed_text = ''.join(filter(lambda char: char in characters, seed_text))
generated_text = generate_text(seed_text, length=200)
print(generated_text)

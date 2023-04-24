import os
import json
import esprima
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow_hub import KerasLayer
import tensorflow_hub as hub

# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# Load dataset
def load_dataset():
    with open('dataset.json', 'r') as f:
        dataset = json.load(f)
    return dataset


def tokenize_code(code):
    try:
        tokens = esprima.tokenize(code)
        return [token.value for token in tokens]
    except Exception as e:
        print(e)
        return []


def create_mappings():
    topic_mapping = {
        "variables": 0,
        "data types": 1,
        "conditionals": 2,
        "loops": 3,
        "functions": 4,
        "arrays": 5,
        "objects": 6,
        "classes": 7,
        "DOM manipulation": 8,
        "AJAX": 9,
        "event handling": 10,
        "error handling": 11,
        "callbacks": 12,
        "promises": 13,
        "async/await": 14,
    }
    type_mapping = {
        "VariableDeclaration": 0,
        "Literal": 1,
        "Identifier": 2,
        "IfStatement": 3,
        "SwitchStatement": 4,
        "ConditionalExpression": 5,
        "ForStatement": 6,
        "WhileStatement": 7,
        "DoWhileStatement": 8,
        "ForInStatement": 9,
        "ForOfStatement": 10,
        "FunctionDeclaration": 11,
        "FunctionExpression": 12,
        "ArrowFunctionExpression": 13,
        "ArrayExpression": 14,
        "ObjectExpression": 15,
        "ClassDeclaration": 16,
        "ClassExpression": 17,
        "CallExpression": 18,
        "TryStatement": 19,
        "NewExpression": 20,
    }
    return topic_mapping, type_mapping

# Create one-hot labels
def create_one_hot_labels(dataset, topic_mapping, type_mapping):
    # One-hot encode labels
    one_hot_labels = []
    for sample in dataset:
        one_hot_topic_label = [0] * len(topic_mapping)
        for topic in sample['labels']['topic']:
            one_hot_topic_label[topic_mapping[topic]] = 1

        one_hot_type_label = [0] * len(type_mapping)
        for type_ in sample['labels']['type']:
            one_hot_type_label[type_mapping[type_]] = 1

    one_hot_labels.append(one_hot_topic_label + one_hot_type_label)
    one_hot_labels = np.array(one_hot_labels)
    return one_hot_labels


# Encode samples
def encode_samples(tokenized_samples):
    # Load Universal Sentence Encoder model
    use_model = KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)
    # Encode samples
    print('Encoding samples...')
    encoded_samples = use_model([tokens.join(' ') for tokens in tokenized_samples])
    print('Encoding samples complete.')
    return encoded_samples
    


def another_function():
    dataset_json = '{"samples": [{"code": "const a = 10;", "labels": {"topic": ["variables"], "type": ["VariableDeclaration"]}}]}'
    train_model_with_dataset(dataset_json)



def train_model_with_dataset(dataset_json):
    dataset = dataset_json
    tokenized_samples = [tokenize_code(sample['code']) for sample in dataset]

    topic_mapping, type_mapping = create_mappings()
    one_hot_labels = create_one_hot_labels(dataset, topic_mapping, type_mapping)
    encoded_samples = encode_samples(tokenized_samples)

    train_split = 0.8
    x_train, x_val, y_train, y_val = train_test_split(encoded_samples, one_hot_labels, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(encoded_samples.shape[1],)),
        layers.Dense(len(topic_mapping) + len(type_mapping), activation='sigmoid')
    ])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    print('Training model...')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
    print('Training model complete.')

    # Make predictions on the validation set
    y_pred = np.round(model.predict(x_val))

    # Calculate metrics
    num_classes = y_val.shape[1]
    metrics = []

    for i in range(num_classes):
        class_accuracy = accuracy_score(y_val[:, i], y_pred[:, i])
        class_precision = precision_score(y_val[:, i], y_pred[:, i], zero_division=0)
        class_recall = recall_score(y_val[:, i], y_pred[:, i], zero_division=0)
        class_f1 = f1_score(y_val[:, i], y_pred[:, i], zero_division=0)

        metrics.append({
            'classIndex': i,
            'accuracy': class_accuracy,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
        })

    print(metrics)


if __name__ == "__main__":
    print('Starting Import of dataset')
    dataset = load_dataset()
    print('Import of dataset success')
    train_model_with_dataset(dataset)
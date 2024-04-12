#!/usr/bin/env python
# coding: utf-8



import os
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Accuracy, F1Score
from ctgan import CTGAN
import sys
sys.path.append( '../util' )
import util as util
import joblib
import numpy as np


def create_multiclass_classification_model(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x) 
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    


synthesizer = CTGAN.load('../generator_custom/medium_synthesizer.pkl')
synthetic_dataset = synthesizer.sample(24000000)

train, test = util.import_dataset(7,"dnn")
del train

discrete_columns = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 
                    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

encoders_new = joblib.load('../data_breakdown/column_encoders.joblib')
for column in discrete_columns:
    if column in encoders_new:
        #####for doing this for the dnn's,do NOT do this for the label column######
        synthetic_dataset[column] = encoders_new[column].inverse_transform(synthetic_dataset[column])

y_train = synthetic_dataset[util.y_column]
y_test = test[util.y_column]

X_train = synthetic_dataset.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = y_train.to_numpy()
y_test_encoded = label_encoder.fit_transform(y_test)

del test
num_classes = len(label_encoder.classes_)
print(" Number of classes is:" )
print(num_classes)
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#https://keras.io/api/models/model_training_apis/
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])


test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

del X_train,y_train




import matplotlib.pyplot as plt
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')
# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy for 7 classes')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('../neuralnetwork/synth_history')




###Change this name so you don't overrwrite the one we have now
model.save('../neuralnetwork/synth_dnn_model.keras')


y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print("Accuracy: ", acc)
print()
f1 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print("F1 Score: ", f1)






from tensorflow.keras.models import load_model
loaded_model = load_model('../neuralnetwork/synth_dnn_model.keras')
print(loaded_model.summary())
print(loaded_model.get_config())




for layer in loaded_model.layers:
    weights = layer.get_weights()  
    #ValueError: not enough values to unpack (expected 2, got 0) <- fixing this error, not all llayers have bias or weight
    if len(weights) > 0:
        print(f"{layer.name} weights shape: {weights[0].shape}")
        if len(weights) > 1:
            print(f"{layer.name} biases shape: {weights[1].shape}")
    else:
        print(f"{layer.name} has no weights or biases.")




from tensorflow.keras.utils import plot_model





plot_model(loaded_model, to_file='../neuralnetwork/synth_model_plot.png', show_shapes=True, show_layer_names=True)





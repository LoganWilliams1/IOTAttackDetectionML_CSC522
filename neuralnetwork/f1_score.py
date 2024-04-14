from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append( '../util' )
import util as util
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




# model = load_model('../neuralnetwork/dnn_model.keras')

## 2 class ##

train, test = util.import_dataset(2,"dnn")
y_train = train[util.y_column]
y_test = test[util.y_column]

X_train = train.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)


del train,test,y_train,y_test
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])

del X_train,y_train_encoded

y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)
del X_test, y_pred

print("2 Classes")
print()
acc_2 = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print()
f1_2 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print()
print()
print()
del y_test_encoded, predictions



## 8 class ##

train, test = util.import_dataset(7,"dnn")
y_train = train[util.y_column]
y_test = test[util.y_column]

X_train = train.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)


del train,test,y_train,y_test
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])

del X_train,y_train_encoded

y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)
del X_test, y_pred

print("8 Classes")
print()
acc_8 = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print()
f1_8 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print()
print()
print()
del y_test_encoded, predictions



## 34 class ##

train, test = util.import_dataset("dnn")
y_train = train[util.y_column]
y_test = test[util.y_column]

X_train = train.drop(util.y_column, axis=1)
X_test = test.drop(util.y_column, axis=1)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)


del train,test,y_train,y_test
model = create_multiclass_classification_model(len(util.X_columns),num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x=X_train, y=y_train_encoded,
                    validation_split=0.2, epochs=100, 
                    batch_size=256, callbacks=[early_stopping])

del X_train,y_train_encoded

y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)
del X_test, y_pred

print("34 Classes")
print()
acc_34 = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print()
f1_34 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print()
print()
print()
del y_test_encoded, predictions



print("2 Classes")
print()
print("Accuracy: ", acc_2)
print("F1 Score: ", f1_2)
print()
print()
print("8 Classes")
print()
print("Accuracy: ", acc_8)
print("F1 Score: ", f1_8)
print()
print()
print("34 Classes")
print()
print("Accuracy: ", acc_34)
print("F1 Score: ", f1_34)
print()
print()

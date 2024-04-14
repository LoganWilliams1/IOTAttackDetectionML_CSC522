from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append( '../util' )
import util as util
import numpy as np




model = load_model('../neuralnetwork/dnn_model.keras')

## 2 class ##

train, test = util.import_dataset(2, "dnn")
del train

y_test = test[util.y_colymn]
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
X_test = test.drop(util.y_column, axis=1)
del test, y_test

y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)
del X_test, y_pred

print("2 Classes")
print()
acc = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print("Accuracy: ", acc)
print()
f1 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print("F1 Score: ", f1)
print()
print()
print()
del y_test_encoded, predictions



## 8 class ##

train, test = util.import_dataset(7, "dnn")
del train

y_test = test[util.y_colymn]
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
X_test = test.drop(util.y_column, axis=1)
del test, y_test

y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)
del X_test, y_pred

print("8 Classes")
print()
acc = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print("Accuracy: ", acc)
print()
f1 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print("F1 Score: ", f1)
print()
print()
print()
del y_test_encoded, predictions



## 34 class ##

train, test = util.import_dataset("dnn")
del train

y_test = test[util.y_colymn]
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
X_test = test.drop(util.y_column, axis=1)
del test, y_test

y_pred = model.predict(X_test, verbose=2)
predictions = np.argmax(y_pred, axis=1)
del X_test, y_pred

print("34 Classes")
print()
acc = accuracy_score(y_true=y_test_encoded, y_pred=predictions)
print("Accuracy: ", acc)
print()
f1 = f1_score(y_true=y_test_encoded, y_pred=predictions, average="macro")
print("F1 Score: ", f1)
print()
print()
print()
del y_test_encoded, predictions


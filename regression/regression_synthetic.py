import sys
sys.path.append( '../util' )
import util as util
import regression
from ctgan import CTGAN
import pandas as pd
from sklearn.preprocessing import LabelEncoder



synthesizer = CTGAN.load('../generator_custom/medium_synthesizer.pkl')
synthetic_dataset = synthesizer.sample(24000000)

train, test = util.import_dataset(7, "regression")
del train


label_encoder = LabelEncoder()
test[util.y_column] = label_encoder.fit_transform(test[util.y_column])


y_pred, y_test = regression.train_test_logistic_regression(synthetic_dataset, test)

y_pred = pd.Categorical(y_pred.flatten())



print("7 Class Synthetic Data LR Classifier")
print()
regression.print_scores(y_pred, y_test)




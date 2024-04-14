import sys
sys.path.append( '../util' )
import util as util
import regression
from ctgan import CTGAN



synthesizer = CTGAN.load('../generator_custom/medium_synthesizer.pkl')
synthetic_dataset = synthesizer.sample(24000000)

train, test = util.import_dataset(7, "regression")
del train


# encoders_new = joblib.load('../data_breakdown/column_encoders.joblib')
# for column in discrete_columns:
#     if column in encoders_new:
#         synthetic_dataset[column] = encoders_new[column].inverse_transform(synthetic_dataset[column])

# y_train = synthetic_dataset[util.y_column]
# y_test = test[util.y_column]

# X_train = synthetic_dataset.drop(util.y_column, axis=1)
# X_test = test.drop(util.y_column, axis=1)

# label_encoder = LabelEncoder()
# y_train_encoded = y_train.to_numpy()
# y_test_encoded = label_encoder.fit_transform(y_test)

# del test
# num_classes = len(label_encoder.classes_)
# print(" Number of classes is:" )
# print(num_classes)


y_pred, y_test = regression.train_test_logistic_regression(synthetic_dataset, test)

regression.print_scores(y_pred, y_test)




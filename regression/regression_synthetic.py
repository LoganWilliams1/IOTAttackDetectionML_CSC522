import sys
sys.path.append( '../util' )
import util as util
import regression
from ctgan import CTGAN



synthesizer = CTGAN.load('../generator_custom/medium_synthesizer.pkl')
synthetic_dataset = synthesizer.sample(24000000)

train, test = util.import_dataset(7, "regression")
del train

synthetic_dataset[util.y_column].to_numpy()

y_pred, y_test = regression.train_test_logistic_regression(synthetic_dataset, test)

print("7 Class Synthetic Data LR Classifier")
print()
regression.print_scores(y_pred, y_test)




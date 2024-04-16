from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata
import sys
sys.path.append( '../util' )
import util as util
import pandas as pd
from ctgan import CTGAN


train, test = util.import_dataset("dnn", subset_frac=0.5)

df = pd.concat([train, test], ignore_index=True)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

synthesizer = CTGAN.load('../generator_custom/medium_synthesizer.pkl')
synthetic_data = synthesizer.sample(24000000)

columns = util.X_columns
columns.append(util.y_column)

for column in columns:
    fig = get_column_plot(
        real_data=df,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=column  
    )

    fig.write_image(f"../data_breakdown/column_dists/{column}.png")  







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

fig = get_column_plot(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='Telnet'
)

fig.write_image("Telnet.png")

fig = get_column_plot(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='IRC'
)

fig.write_image("IRC.png")

fig = get_column_plot(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='Header_Length'
)

fig.write_image("Header_Length.png")

fig = get_column_plot(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='Drate'
)

fig.write_image("Drate.png")

fig = get_column_plot(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='syn_flag_number'
)

fig.write_image("syn_flag_number.png")




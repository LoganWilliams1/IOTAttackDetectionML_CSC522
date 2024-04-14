#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append( '../util' )
import util as util
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer


# In[2]:


# creates and checks metadata object

def get_metadata(train):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    metadata_dict = metadata.to_dict()
    print()
    print("METADATA:")
    for key, value in metadata_dict.items():
        print(key, value)
    print()
    print("COLUMNS:")
    columns = metadata_dict.get('columns')
    for key, value in columns.items():
        sdtype = value.get('sdtype')
        print(key + ": " + sdtype)
    
    return metadata


# In[5]:


def get_synthesizer(stype, name):
    # train/test split, get metadata for train
    train, test = util.import_dataset(subset_frac=0.5)
    metadata = get_metadata(train)
    synthesizer = None
    filepath = "../synthetic_data/synthesizer_pickles/" + name


    if stype == "fml":
        # make and fit fast synthesizer
        synthesizer = SingleTablePreset(metadata, name='FAST_ML')
        synthesizer.fit(train) 

    elif stype == "gcs":
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(train)

    elif stype == "gan":
        synthesizer = CTGANSynthesizer(metadata)
        synthesizer.fit(train)


    try: 
        synthesizer.save(filepath=filepath)
    except Exception as e:
        print(f"Error occurred while saving file: {e}")
    del train
    
    return synthesizer, test


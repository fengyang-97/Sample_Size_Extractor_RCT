import os 
import numpy as np
import pandas as pd
import pickle
import sys

from utils import Sample_Size_Extractor
from utils import performance

import timeit
from sklearn.utils import shuffle
#from sklearn.model_selection import KFold



max_features=10000
epochs=10 
batch_size=32
wvs = Sample_Size_Extractor.load_trained_w2v_model("PubMed-w2v.bin")

# Train with all data and save weights
# load dataframe: 400 abstracts from PubMed, Covid-Set(300) + General-Set(100)
# df_all = pd.read_csv('data/df_ab_w_tt_ss_400.csv')
# preprocessor = Sample_Size_Extractor.Preprocessor(max_features, wvs, df_all["abstract"].values)
# X_tr, y_tr = Sample_Size_Extractor.generate_X_y(df_all)
# nn_ = Sample_Size_Extractor.SampleSizeClassifier(preprocessor)
# nn_.fit_MLP_model()
# nn_.model.load_weights(pre_w_path)
# X_tr_fvs = nn_.featurize_for_input(X_tr)
# nn_.model.fit(X_tr_fvs, y_tr, epochs=10, batch_size=32, verbose=0)
# pmid_list_, ss_pred_list_, conf_pred_list_ = performance.get_test_result(df_all, nn_, 0.2)
# df_test_pred = pd.DataFrame({'pmid':pmid_list_, 'pred_ss':ss_pred_list_, 'conf': conf_pred_list_})
# df_test_pred.to_csv('data/result/pred_pretraining_400_nocv.csv')
# nn_.model.save_weights('data/pretrained_weights/SSE_pretraining_with_ebmnlp.h5')



# Use our pretrained Sample Size Extractor
if __name__ == '__main__':
    df_to_extract = pd.read_csv('data/sample_input.csv')
    # load all text in pretrained abstract for initializing the preprocessor
    with open('data/400_abstract_text.pickle', 'rb') as handle:
        prev_abstract_text = pickle.load(handle)

    # Use our sample size extractor
    pre_w_path = 'data/pretrained_weights/SSE_pretraining_with_ebmnlp.h5'
    max_features=10000
    epochs=10 
    batch_size=32

    #wvs was loaded before
    #wvs = Sample_Size_Extractor.load_trained_w2v_model("PubMed-w2v.bin")
    p_all_text = list(prev_abstract_text)+list(df_to_extract["abstract"].values)
    preprocessor = Sample_Size_Extractor.Preprocessor(max_features, wvs, p_all_text)

    nn_ = Sample_Size_Extractor.SampleSizeClassifier(preprocessor)
    # load our trained model
    nn_.fit_MLP_model()
    nn_.model.load_weights(pre_w_path)
    pmid_list_, ss_pred_list_, conf_pred_list_ = performance.get_test_result(df_to_extract, nn_, 0.2)
    df_test_pred = pd.DataFrame({'pmid':pmid_list_, 'pred_ss':ss_pred_list_, 'conf': conf_pred_list_})
    df_test_pred.to_csv('data/sample_output.csv', index = False)
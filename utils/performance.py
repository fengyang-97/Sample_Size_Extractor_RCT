import spacy
#nlp = spacy.load('en')
nlp = spacy.load("en_core_web_sm")

import pandas as pd
import numpy as np

def predict_sample_size_result(nn_model, abstract, threshold):
    sample_size = None
    confidence = None
    sample_size_pred = nn_model.predict_for_abstract(abstract)  
    if sample_size_pred is not None:
        n, confidence = sample_size_pred
        if confidence >= threshold:
        #sample_size_str = n
            sample_size = int(n)
            confidence = confidence[0]
    return sample_size, confidence

def get_test_result(df_test, nn_model, threshold):
    pmid_list = []
    ss_pred_list = []
    conf_pred_list = []
    for instance in df_test.iterrows():
        instance = instance[1]
        cur_ab = nlp(instance["abstract"]).text
        ss_pred, conf_pred = predict_sample_size_result(nn_model, cur_ab, threshold)
        pmid_list.append(instance["pmid"])
        ss_pred_list.append(ss_pred)
        conf_pred_list.append(conf_pred)
    return pmid_list, ss_pred_list, conf_pred_list

def compare_result(x):
    output = False
    if x['tt_sample_size'] == x['pred_ss']:
        output = True
    elif x['no_ss'] == x['no_pred_ss'] == True:
        output = True
    
    return output

def calculate_performance_exact(df_test, df_test_pred):
    # merge
    df_test1 = df_test[['pmid','tt_sample_size']]
    df_test1['pmid'] = df_test1['pmid'].apply(lambda x: int(x))
    df_test_pred['pmid'] = df_test_pred['pmid'].apply(lambda x: int(x))
    tt_ss_merge = df_test1.merge(df_test_pred, how='outer', on='pmid')
    # mark na
    tt_ss_merge['no_ss'] = tt_ss_merge['tt_sample_size'].apply(pd.isna)
    tt_ss_merge['no_pred_ss'] = tt_ss_merge['pred_ss'].apply(pd.isna)
    
    tt_ss_merge['same'] = tt_ss_merge.apply(compare_result, axis = 1)
    
    TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)])
    #TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)&
    #         (tt_ss_merge['no_pred_ss']==False)])

    #TN = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==True)])
    TN = len(tt_ss_merge[(tt_ss_merge['no_ss']==True) & (tt_ss_merge['no_pred_ss']==True)])

    FP = len(tt_ss_merge[(tt_ss_merge['same']==False) & (tt_ss_merge['no_pred_ss']==False)])

    FN = len(tt_ss_merge[(tt_ss_merge['no_pred_ss']==True) & (tt_ss_merge['no_ss']==False)])

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/len(df_test)
    #acc2 = len(tt_ss_merge[(tt_ss_merge['same']==True)])/len(df_test)
    f1 = 2*(precision*recall)/(precision+recall)
    
    return [precision, recall, acc, f1]

def calculate_performance_exact_cv(df_test, df_test_pred):
    perf_fold_dict = {}
    # merge
    df_test1 = df_test[['pmid','tt_sample_size','fold_id']]
    df_test1['pmid'] = df_test1['pmid'].apply(lambda x: int(x))
    df_test_pred['pmid'] = df_test_pred['pmid'].apply(lambda x: int(x))
    tt_ss_merge = df_test1.merge(df_test_pred, how='outer', on='pmid')
    # mark na
    tt_ss_merge['no_ss'] = tt_ss_merge['tt_sample_size'].apply(pd.isna)
    tt_ss_merge['no_pred_ss'] = tt_ss_merge['pred_ss'].apply(pd.isna)
    
    tt_ss_merge['same'] = tt_ss_merge.apply(compare_result, axis = 1)
    
    for fold_id in list(set(tt_ss_merge['fold_id'])):
        cur_fold_df = tt_ss_merge[tt_ss_merge['fold_id'] == fold_id]
        TP = len(cur_fold_df[(cur_fold_df['same']==True) & (cur_fold_df['no_ss']==False)])
        #TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)&
        #         (tt_ss_merge['no_pred_ss']==False)])

        #TN = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==True)])
        TN = len(cur_fold_df[(cur_fold_df['no_ss']==True) & (cur_fold_df['no_pred_ss']==True)])

        FP = len(cur_fold_df[(cur_fold_df['same']==False) & (cur_fold_df['no_pred_ss']==False)])

        FN = len(cur_fold_df[(cur_fold_df['no_pred_ss']==True) & (cur_fold_df['no_ss']==False)])

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        acc = (TP+TN)/len(cur_fold_df)
        #acc2 = len(tt_ss_merge[(tt_ss_merge['same']==True)])/len(df_test)
        f1 = 2*(precision*recall)/(precision+recall)
        
        perf_fold_dict[fold_id] = [precision, recall, acc, f1]
    
    return perf_fold_dict

def loose_match(pmid, tt_pred, ann_dict, tor_perc):
    """a loose version for compare_result()"""
    pmid = str(int(pmid))
    ann_item = ann_dict[pmid]
    match = False
    if 'Poss_total_sample' in ann_item.keys():
        poss_ls = ann_item['Poss_total_sample']
        poss_set = set(poss_ls) - set([None]) # remove None
        if tt_pred in poss_set:
            match = True
    
    if 'Total_sample_size' in ann_item.keys():
        tt_ss = ann_item['Total_sample_size']
        if tt_ss == tt_pred:
            match = True
        elif pd.isna(tt_ss)==False & pd.isna(tt_pred) == False:
            if tt_ss*(1-tor_perc)<= tt_pred <=tt_ss*(1+tor_perc):
                match = True
    
    if pd.isna(tt_ss) == True & pd.isna(tt_pred) == True:
        match = True
    
    return match


def calculate_performance_loose(df_test, df_test_pred, tt_poss_dict):
    # merge
    df_test1 = df_test[['pmid','tt_sample_size']]
    df_test1['pmid'] = df_test1['pmid'].apply(lambda x: int(x))
    df_test_pred['pmid'] = df_test_pred['pmid'].apply(lambda x: int(x))
    tt_ss_merge = df_test1.merge(df_test_pred, how='outer', on='pmid')
    # mark na
    tt_ss_merge['no_ss'] = tt_ss_merge['tt_sample_size'].apply(pd.isna)
    tt_ss_merge['no_pred_ss'] = tt_ss_merge['pred_ss'].apply(pd.isna)
    
    tt_ss_merge['same'] = tt_ss_merge.apply(lambda x: loose_match(x['pmid'], x['pred_ss'], 
                                                                      tt_poss_dict, 0.1), axis = 1)
    
    TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)])
    #TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)&
    #         (tt_ss_merge['no_pred_ss']==False)])

    #TN = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==True)])
    TN = len(tt_ss_merge[(tt_ss_merge['no_ss']==True) & (tt_ss_merge['no_pred_ss']==True)])

    FP = len(tt_ss_merge[(tt_ss_merge['same']==False) & (tt_ss_merge['no_pred_ss']==False)])

    FN = len(tt_ss_merge[(tt_ss_merge['no_pred_ss']==True) & (tt_ss_merge['no_ss']==False)])

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/len(df_test)
    #acc2 = len(tt_ss_merge[(tt_ss_merge['same']==True)])/len(df_test)
    f1 = 2*(precision*recall)/(precision+recall)
    
    return [precision, recall, acc, f1]
def calculate_performance_loose_cv(df_test, df_test_pred, tt_poss_dict):
    """
    df_test: gold_w_foldid_df: ['pmid', 'tt_sample_size', 'fold_id']
    """
    # merge
    df_test1 = df_test[['pmid','tt_sample_size', 'fold_id']]
    df_test1['pmid'] = df_test1['pmid'].apply(lambda x: int(x))
    df_test_pred['pmid'] = df_test_pred['pmid'].apply(lambda x: int(x))
    tt_ss_merge = df_test1.merge(df_test_pred, how='outer', on='pmid')
    # mark na
    tt_ss_merge['no_ss'] = tt_ss_merge['tt_sample_size'].apply(pd.isna)
    tt_ss_merge['no_pred_ss'] = tt_ss_merge['pred_ss'].apply(pd.isna)
    
    tt_ss_merge['same'] = tt_ss_merge.apply(lambda x: loose_match(x['pmid'], x['pred_ss'], 
                                                                      tt_poss_dict, 0.1), axis = 1)
    
    perf_fold_dict = {}
    #print(tt_ss_merge.columns)
    #print(list(set(tt_ss_merge['fold_id'])))
    for fold_id in list(set(tt_ss_merge['fold_id'])):
        cur_fold_df = tt_ss_merge[tt_ss_merge['fold_id'] == fold_id]
        TP = len(cur_fold_df[(cur_fold_df['same']==True) & (cur_fold_df['no_ss']==False)])
        #TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)&
        #         (tt_ss_merge['no_pred_ss']==False)])

        #TN = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==True)])
        TN = len(cur_fold_df[(cur_fold_df['no_ss']==True) & (cur_fold_df['no_pred_ss']==True)])

        FP = len(cur_fold_df[(cur_fold_df['same']==False) & (cur_fold_df['no_pred_ss']==False)])

        FN = len(cur_fold_df[(cur_fold_df['no_pred_ss']==True) & (cur_fold_df['no_ss']==False)])

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        acc = (TP+TN)/len(cur_fold_df)
        #acc2 = len(tt_ss_merge[(tt_ss_merge['same']==True)])/len(df_test)
        f1 = 2*(precision*recall)/(precision+recall)
        
        perf_fold_dict[fold_id] = [precision, recall, acc, f1]
    return perf_fold_dict

def calculate_performance_exact_but_not_all(df_test, df_test_pred):
    # merge
    df_test1 = df_test[['pmid','tt_sample_size']]
    df_test1['pmid'] = df_test1['pmid'].apply(lambda x: int(x))
    df_test_pred['pmid'] = df_test_pred['pmid'].apply(lambda x: int(x))
    tt_ss_merge = df_test1.merge(df_test_pred, how='inner', on='pmid')
    # mark na
    tt_ss_merge['no_ss'] = tt_ss_merge['tt_sample_size'].apply(pd.isna)
    tt_ss_merge['no_pred_ss'] = tt_ss_merge['pred_ss'].apply(pd.isna)
    
    tt_ss_merge['same'] = tt_ss_merge.apply(compare_result, axis = 1)
    
    TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)])
    #TP = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==False)&
    #         (tt_ss_merge['no_pred_ss']==False)])

    #TN = len(tt_ss_merge[(tt_ss_merge['same']==True) & (tt_ss_merge['no_ss']==True)])
    TN = len(tt_ss_merge[(tt_ss_merge['no_ss']==True) & (tt_ss_merge['no_pred_ss']==True)])

    FP = len(tt_ss_merge[(tt_ss_merge['same']==False) & (tt_ss_merge['no_pred_ss']==False)])

    FN = len(tt_ss_merge[(tt_ss_merge['no_pred_ss']==True) & (tt_ss_merge['no_ss']==False)])

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/len(df_test)
    #acc2 = len(tt_ss_merge[(tt_ss_merge['same']==True)])/len(df_test)
    f1 = 2*(precision*recall)/(precision+recall)
    
    return [precision, recall, acc, f1]
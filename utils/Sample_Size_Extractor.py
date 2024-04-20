import numpy as np
from utils import index_numbers

import gensim
#import tensorflow.keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, LeakyReLU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer

import spacy
nlp = spacy.load("en_core_web_sm")

def replace_n_equals(abstract_tokens):
    """
    :param abstract_tokens:
    :return: abstract tokens with all "n=" replaced with ""
    """
    for j, t in enumerate(abstract_tokens):
        if "n=" in t.lower():
            # special case for sample size reporting
            t_n = t.split("=")[1].replace(")", "")
            abstract_tokens[j] = t_n
    return abstract_tokens

def tokenize_abstract(abstract, nlp=None):
    """
    :param abstract: abstract (text)
    :return: tokens(list), POS_tags(list)
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    tokens, POS_tags = [], []
    ab = nlp(abstract)
    for word in ab:
        tokens.append(word.text)
        POS_tags.append(word.tag_)

    return tokens, POS_tags

def check_is_int(w):
    """
    Check if the given word is an integer.
    :param s:
    :return:
    """
    try:
        int(w)
        return True
    except:
        return False

def get_window_indices(all_tokens, i, window_size):
    lower_idx = max(0, i-window_size)
    upper_idx = min(i+window_size, len(all_tokens)-1)
    return (lower_idx, upper_idx)

def check_sum(target_num, all_nums_in_abstract):
    """
    Check if the target number is the sum of any two of the number in the abstracts.
    :param target_num:
    :param all_nums_in_abstract:
    :return:
    """
    flag = False
    all_sum_ls = []
    for i, num in enumerate(all_nums_in_abstract[:-1]):
        cur_sum = list(np.array([num]) + np.array(all_nums_in_abstract[i+1:]))
        all_sum_ls = all_sum_ls + cur_sum
    if target_num in all_sum_ls:
        flag = True
    return flag

def word_to_features(abstract_tokens, POS_tags, i,
                    all_nums_in_abstract,
                    years_indices, patient_indices, 
                    patient_indices2, enroll_indices, analyse_indices,
                    window_size_for_years=5,
                    window_size_patient_mention=4,
                    window_size_verb_mention=5):
    """
    Generate a set of features for the given word(should be an integer) based on our designed rule,
    including features about the left/right words, keywords about sample size, numeric features of the given number.

    :param abstract_tokens: a list of all tokens in the abstract
    :param POS_tags: a list of all POS_tag for words in the abstract
    :param i: the index of the given word
    :param all_nums_in_abstract: a list of all number in the abstract
    :param years_indices: a list of indices of word "year"
    :param patient_indices: a list of indices of word in ["patients", "subjects", "participants"]
    :param patient_indices2: a list of indices of word in ["population", "size", "people", "individuals", "adults",
                        "outpatients", "volunteers", "respondents", "providers"]
    :param enroll_indices: a list of indices of word in ["enroll", "enrolled", "recruit", "recruited", "randomized", "screen", "screened"]
    :param analyse_indices:  a list of indices of word in ["assessment", "assessments","analysis", "analysed", "analyzed", "completed",
                       "ITT", "intention-to-treat", "intention"]
    :param window_size_for_years:
    :param window_size_patient_mention:
    :param window_size_verb_mention:
    :return:
        "left_word":[ll_word, l_word],
        "right_word":[rr_word, r_word],
        "left_PoS":l_POS, "right_PoS":r_POS,
        "other_features":[biggest_num_in_abstract, years_mention_within_window,
                                target_looks_like_a_year,
                                patients_mention_follows_within_window,
                             patients_mention_follows_within_window2,
                             enroll_mention_within_window,
                             analyze_mention_within_window,
                             sum_of_two_nums, appear_more_than_once,
                             rank_in_num_prop,
    """
    ll_word, l_word, r_word, rr_word = "", "", "", ""

    l_POS, r_POS   = "", ""
    t_word = abstract_tokens[i]

    if i > 1:
        ll_word = abstract_tokens[i-2].lower()
    else:
        ll_word = "BoS"

    if i > 0:
        l_word = abstract_tokens[i-1].lower()
        l_POS  = POS_tags[i-1]
    else:
        l_word = "BoS"
        l_POS  = "XX" # i.e., unknown

    if i < len(abstract_tokens)-2:
        rr_word = abstract_tokens[i+2].lower()
    else:
        r_word = "LoS"

    if i < len(abstract_tokens)-1:
        r_word = abstract_tokens[i+1].lower()
        r_POS  = POS_tags[i+1]
    else:
        r_word = "LoS"
        r_POS  = "XX"

    target_num = int(t_word) # the word should be an integer
    biggest_num_in_abstract = 0.0 # check if the number is the biggest number in the abstract
    if target_num >= max(all_nums_in_abstract):
        biggest_num_in_abstract = 1.0

    # this feature encodes whether "year" or "years" is mentioned
    # within window_size_for_years tokens of the target (i)
    years_mention_within_window = 0.0
    lower_idx, upper_idx = get_window_indices(abstract_tokens, i, window_size_for_years)
    for year_idx in years_indices:
        if lower_idx < year_idx <= upper_idx:
            years_mention_within_window = 1.0
            break

    # check if word in ["patients", "subjects", "participants"] is mentioned within the window
    patients_mention_follows_within_window = 0.0
    _, upper_idx = get_window_indices(abstract_tokens, i, window_size_patient_mention)
    for patient_idx in patient_indices:
        if i < patient_idx <= upper_idx:
            patients_mention_follows_within_window = 1.0
            break

    target_looks_like_a_year = 0.0
    lower_year, upper_year = 1940, 2022 #check if the number is within range(1940, 2022)
    if lower_year <= target_num <= upper_year:
        target_looks_like_a_year = 1.0
    
    # check if word in ["population", "size", "people", "individuals", "adults",
    #                         "outpatients", "volunteers", "respondents", "providers"] is mentioned within the window
    patients_mention_follows_within_window2 = 0.0
    _, upper_idx = get_window_indices(abstract_tokens, i, window_size_patient_mention)
    for patient_idx in patient_indices2:
        if i < patient_idx <= upper_idx:
            patients_mention_follows_within_window2 = 1.0
            break

    # check if word in ["enroll", "enrolled", "recruit", "recruited", "randomized", "screen", "screened"] is mentioned within the window
    enroll_mention_within_window = 0.0
    lower_idx, upper_idx = get_window_indices(abstract_tokens, i, window_size_verb_mention)
    for enroll_idx in enroll_indices:
        if lower_idx < enroll_idx <= upper_idx:
            enroll_mention_within_window = 1.0
            break

    # check if word in ["assessment", "assessments","analysis", "analysed", "analyzed", "completed",
    #                        "ITT", "intention-to-treat", "intention"] is mentioned within the window
    analyze_mention_within_window = 0.0
    lower_idx, upper_idx = get_window_indices(abstract_tokens, i, window_size_verb_mention)
    for analyze_idx in analyse_indices:
        if lower_idx < analyze_idx <= upper_idx:
            analyze_mention_within_window = 1.0
            break
            
    # check if the number is the sum of any of the two other numbers in the abstract
    sum_of_two_nums = 0.0
    if check_sum(target_num, all_nums_in_abstract):
        sum_of_two_nums = 1.0
    
    appear_more_than_once = int(all_nums_in_abstract.count(target_num)>1)
    # try to add some features about the statistics of the number, depreciated now
    #- largest/second largest
    #- Maximum candidate sample size/sum of candidate sample sizes 
    #- Maximum candidate sample size/minimum candidate sample size 
    #- (Maximum candidate sample size – second largest candidate sample size)/mean of all candidate sample sizes, excluding maximum candidate sample size 
    #- Mean of candidate sample sizes/mean of all candidate sample sizes, excluding maximum candidate sample size 
    #r_max = target_num/max(max(all_nums_in_abstract),1)
    #max_2 = [sorted(all_nums_in_abstract)[1] if len(all_nums_in_abstract)>1 else all_nums_in_abstract[0]][0]
    #r_2ndmax = target_num/max(max_2, 1)
    #r_mean = target_num/max(1,np.mean(all_nums_in_abstract))
    #r_med = target_num/max(1,np.median(all_nums_in_abstract))
    # check the rank(proportion) of the given number among all the numbers in abstract
    rank_in_num_prop = sorted(all_nums_in_abstract).index(target_num)/len(all_nums_in_abstract)
    
    return {"left_word":[ll_word, l_word],
            "right_word":[rr_word, r_word],
            "left_PoS":l_POS, "right_PoS":r_POS,
            "other_features":[biggest_num_in_abstract, years_mention_within_window,
                                target_looks_like_a_year,
                                patients_mention_follows_within_window,
                             patients_mention_follows_within_window2,
                             enroll_mention_within_window,
                             analyze_mention_within_window,
                             #new stat feature
                             sum_of_two_nums, appear_more_than_once,
                             #r_max, 
                             rank_in_num_prop, #r_2ndmax, r_mean, r_med 
                             ]}

def abstract_to_features(abstract_tokens, POS_tags):
    """
    Generate a set of features for all tokens in the abstract.
    :param abstract_tokens:
    :param POS_tags:
    :return: A set of features of all intergers in the abstract.
    """

    years_tokens = ["years", "year"]
    patients_tokens = ["patients", "subjects", "participants"]
    patients_tokens2 = ["population", "size", "people", "individuals", "adults", 
                        "outpatients", "volunteers", "respondents", "providers"]
    enroll_tokens = ["enroll", "enrolled", "recruit", "recruited", "randomized", "screen", "screened"]
    analyse_tokens = ["assessment", "assessments","analysis", "analysed", "analyzed", "completed", 
                       "ITT", "intention-to-treat", "intention"]
    all_nums_in_abstract, years_indices, patient_indices, patient_indices2 = [], [], [], []
    enroll_indices, analyse_indices = [], []
    
    for idx, t in enumerate(abstract_tokens):
        t_lower = t.lower()

        if t_lower in years_tokens:
            years_indices.append(idx)

        if t_lower in patients_tokens:
            patient_indices.append(idx)
        
        if t_lower in patients_tokens2:
            patient_indices2.append(idx)
        
        if t_lower in enroll_tokens:
            enroll_indices.append(idx)
        
        if t_lower in analyse_tokens:
            analyse_indices.append(idx)
            
        try:
            num = int(t)
            all_nums_in_abstract.append(num)
        except:
            pass

    # note that we keep track of all candidates/numbers
    # and pass this back.
    x, numeric_token_indices = [], []
    for word_idx in range(len(abstract_tokens)):
        if (check_is_int(abstract_tokens[word_idx])):
            numeric_token_indices.append(word_idx)
            features = word_to_features(abstract_tokens, POS_tags, word_idx, all_nums_in_abstract,
                                      years_indices, patient_indices, 
                                      patient_indices2, enroll_indices, analyse_indices)
            x.append(features)

    return x, numeric_token_indices

def y_to_bin(y):
    y_bin = np.zeros(len(y))
    for idx, y_i in enumerate(y):
        if y_i == "N":
            y_bin[idx] = 1.0
    return y_bin

def mark(tokenized_abstract, nums_to_labels):
    """

    :param tokenized_abstract:
    :param nums_to_labels: dictionary mapping numbers to labels
    :return: a list of label in the end ["N", "O"], indicating whether the token is labeled with "N" or "0"
    """
    y = []
    for t in tokenized_abstract:

        try:
            t_num = int(t)

            if str(t_num) in nums_to_labels.keys(): # if the number is keys for num_to_labels: the number is N or n1 or n2
                y.append(nums_to_labels[str(t_num)]) # if yes, append the label to y or "O"
            else:
                y.append("O")
        except:
            y.append("O")
    return y


def generate_X_y(df):
    """
    Generate the X, y for model training and testing
    :param df: a dataframe with all features and y
    :return: X, y for model training and testing
    """

    X, y_labels = [], []
    for item in df.iterrows():
        item = item[1]

        abstract_tokens, POS_tags = tokenize_abstract(item["abstract"], nlp)
        abstract_tokens = replace_n_equals(abstract_tokens)
        
        tt_ss = item["tt_sample_size"]
        try:
            tt_ss_int = int(tt_ss)
            tt_ss2 = str(tt_ss_int)
        except:
            tt_ss2 = str(tt_ss)
           
        nums_to_labels = {tt_ss2:"N"}
        cur_y = mark(abstract_tokens, nums_to_labels) #find the idx for the sample size
        cur_x, numeric_token_indices = abstract_to_features(abstract_tokens, POS_tags)

        X.extend(cur_x)
        y_labels.extend([cur_y[idx] for idx in numeric_token_indices]) #cur_y[idx]: the label for the current idx
        # y_to_bin(y): [0,0,0,1,0,0,0]: 1 if the current number is "N" the total sample size
    
    y = y_to_bin(y_labels)

    return X, y

def generate_X(df):
    """
    Generate the X for fitting the model
    :param df: a dataframe with abstract and pmid
    :return: X for model fitting
    """

    X= []
    for item in df.iterrows():
        item = item[1]

        abstract_tokens, POS_tags = tokenize_abstract(item["abstract"], nlp)
        abstract_tokens = replace_n_equals(abstract_tokens)

        cur_x, numeric_token_indices = abstract_to_features(abstract_tokens, POS_tags)

        X.extend(cur_x)

    return X

def load_trained_w2v_model(path):
    m = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return m

class Preprocessor:
    """
    A preprocessor for abstract
    """
    def __init__(self, max_features, wvs, all_texts, unk=True, unk_symbol="unkunk"):
        '''
        max_features: the upper bound to be placed on the vocabulary size.
        embedding_dims: size of the token embeddings; over-ridden if pre-trained
                          vectors is provided (if wvs is not None).
        wvs: set of word vectors to be used for initialization
        '''
        self.unk = unk
        self.unk_symbol = unk_symbol
        self.max_features = max_features
        self.tokenizer = Tokenizer(nb_words=self.max_features)

        self.embedding_dims = wvs.vector_size
        self.word_embeddings = wvs

        self.raw_texts = all_texts
        self.unked_texts = []
        self.fit_tokenizer()

        if self.unk:
            # rewrite the 'raw texts' with unked versions, where tokens not in the
            # top max_features are unked.
            sorted_tokens = sorted(self.tokenizer.word_index, key=self.tokenizer.word_index.get)
            self.known_tokens = sorted_tokens[:self.max_features]
            self.tokens_to_unk = sorted_tokens[self.max_features:]

            for idx, text in enumerate(self.raw_texts):
                cur_text = text_to_word_sequence(text, split=self.tokenizer.split)
                t_or_unk = lambda t : t if t in self.known_tokens else self.unk_symbol
                unked_text = [t_or_unk(t) for t in cur_text]
                unked_text = self.tokenizer.split.join(unked_text)

                self.unked_texts.append(unked_text)

            self.raw_texts = self.unked_texts
            self.fit_tokenizer()

        self.init_word_vectors()


    def fit_tokenizer(self):
        ''' Fits tokenizer to all raw texts; remembers indices->words mappings. '''
        self.tokenizer.fit_on_texts(self.raw_texts)
        self.word_indices_to_words = {}
        for token, idx in self.tokenizer.word_index.items():
            self.word_indices_to_words[idx] = token

    def init_word_vectors(self):
        '''
        Initialize word vectors.
        '''
        self.init_vectors = []
        unknown_words_to_vecs = {}
        for t, token_idx in self.tokenizer.word_index.items():
            if token_idx <= self.max_features:
                try:
                    self.init_vectors.append(self.word_embeddings[t])
                except:
                    if t not in unknown_words_to_vecs:
                        # randomly initialize
                        unknown_words_to_vecs[t] = np.random.random(
                                                self.embedding_dims)*-2 + 1

                    self.init_vectors.append(unknown_words_to_vecs[t])



        self.init_vectors = [np.vstack(self.init_vectors)]

class SampleSizeClassifier:

    def __init__(self, preprocessor, magic_threshold=None):
        self.preprocessor = preprocessor
        self.nlp = spacy.load("en_core_web_sm")
        self.PoS_tags_to_indices = {}
        # all the POS tags are defined as the one in spacy package
        self.tag_names = [u'""', u'#', u'$', u"''", u',', u'-LRB-', u'-RRB-', u'.', u':', u'ADD', u'AFX', u'BES', u'CC', u'CD', u'DT', u'EX', u'FW', u'GW', u'HVS', u'HYPH', u'IN', u'JJ', u'JJR', u'JJS', u'LS', u'MD', u'NFP', u'NIL', u'NN', u'NNP', u'NNPS', u'NNS', u'PDT', u'POS', u'PRP', u'PRP$', u'RB', u'RBR', u'RBS', u'RP', u'SP', u'SYM', u'TO', u'UH', u'VB', u'VBD', u'VBG', u'VBN', u'VBP', u'VBZ', u'WDT', u'WP', u'WP$', u'WRB', u'XX', u'``']
        for idx, tag in enumerate(self.tag_names):
            self.PoS_tags_to_indices[tag] = idx

        self.n_tags = len(self.tag_names)

        # threshold to decide whether the predicted number is a total sample size
        # only number with a confidence greater with 0.25 will be considered (when threshold not defined)
        if magic_threshold is None:
            self.magic_threshold = 0.25

        else:
            self.magic_threshold = magic_threshold

        self.number_tagger = index_numbers.NumberTagger()


    def PoS_tags_to_one_hot(self, tag):
        """
        Transform the list of POS tags to a one hot matrix for training.
        """
        one_hot = np.zeros(self.n_tags)
        if tag in self.PoS_tags_to_indices:
            one_hot[self.PoS_tags_to_indices[tag]] = 1.0
        else:
            pass
        return one_hot

    def featurize_for_input(self, X):
        """
        Generate a X dictionary for model training and testing.
        :param X: abstract_to_features(abstract_tokens, POS_tags)
        """
        left_token_inputs, left_PoS, right_token_inputs, right_PoS, other_inputs = [], [], [], [], []#, []

        # helper func for looking up word indices
        def get_w_index(w):
            #unk_idx = self.preprocessor.tokenizer.word_index[self.preprocessor.unk_symbol]
            if self.preprocessor.unk_symbol in self.preprocessor.tokenizer.word_index.keys():
                unk_idx = self.preprocessor.tokenizer.word_index[self.preprocessor.unk_symbol]
            else:
                unk_idx = max(self.preprocessor.tokenizer.word_index.values())+1
            try:
                word_idx = self.preprocessor.tokenizer.word_index[w]
                if word_idx < self.preprocessor.max_features:
                    return word_idx
                else:
                    return unk_idx
            except:
                pass

            return unk_idx



        for x in X:
            l_word_idx = np.array([get_w_index(w_i) for w_i in x["left_word"]])
            left_token_inputs.append(np.array([l_word_idx]))

            left_PoS.append(self.PoS_tags_to_one_hot(x["left_PoS"]))

            r_word_idx = np.array([get_w_index(w_i) for w_i in x["right_word"]])
            right_token_inputs.append(np.array(r_word_idx))

            right_PoS.append(self.PoS_tags_to_one_hot(x["right_PoS"]))

            other_inputs.append(np.array(x["other_features"]))


        X_inputs_dict = {"left_token_input":np.vstack(left_token_inputs),
                        "left_PoS_input":np.vstack(left_PoS),
                        "right_token_input":np.vstack(right_token_inputs),
                        "right_PoS_input":np.vstack(right_PoS),
                        "other_feature_inputs":np.vstack(other_inputs)}

        return X_inputs_dict


    def fit_MLP_model(self):
        NUM_WINDOW_FEATURES = 2
        left_token_input = Input(name='left_token_input', shape=(NUM_WINDOW_FEATURES,))
        left_token_embedding = Embedding(output_dim=self.preprocessor.embedding_dims, input_dim=self.preprocessor.max_features,
                                        input_length=NUM_WINDOW_FEATURES)(left_token_input)
        left_token_embedding = Flatten(name="left_token_embedding")(left_token_embedding)

        n_PoS_tags = len(self.tag_names)
        left_PoS_input = Input(name='left_PoS_input', shape=(n_PoS_tags,))

        right_token_input = Input(name='right_token_input', shape=(NUM_WINDOW_FEATURES,))
        right_token_embedding = Embedding(output_dim=self.preprocessor.embedding_dims, input_dim=self.preprocessor.max_features,
                                          input_length=NUM_WINDOW_FEATURES)(right_token_input)
        right_PoS_input = Input(name='right_PoS_input', shape=(n_PoS_tags,))

        right_token_embedding = Flatten(name="right_token_embedding")(right_token_embedding)

        other_features_input = Input(name='other_feature_inputs', shape=(10,))

        x = Concatenate(axis=1)([left_token_embedding,
                    right_token_embedding,
                    left_PoS_input, right_PoS_input, other_features_input])
        x = Dense(1028)(x)
        x = LeakyReLU()(x)
        x = Dense(512)(x)
        x = LeakyReLU()(x)
        x = Dense(128, name="hidden1", activation='relu')(x)
        x = Dropout(.2)(x)
        x = Dense(64, name="hidden2", activation='relu')(x)

        # use sigmoid activation to generate the confidence for each number in the abstract
        output = Dense(1, name="prediction", activation='sigmoid')(x)

        self.model = Model([left_token_input, left_PoS_input,
                            right_token_input, right_PoS_input, other_features_input],
                           outputs=[output])

        self.model.compile(optimizer="adam", loss="binary_crossentropy")


    def predict_for_abstract(self, abstract_text):
        '''
        Generate the predicted total sample size from the abstract.
        Return an integer or None if no sample size could be extracted.
        '''
        abstract_text_w_numbers = self.number_tagger.swap(abstract_text)
        abstract_tokens, POS_tags = tokenize_abstract(abstract_text_w_numbers, self.nlp)

        abstract_tokens = replace_n_equals(abstract_tokens)

        if not any((check_is_int(t) for t in abstract_tokens)):
            return None

        abstract_features, numeric_token_indices = abstract_to_features(abstract_tokens, POS_tags)


        X = self.featurize_for_input(abstract_features)
        preds = self.model.predict(X)
        most_likely_idx = np.argmax(preds)

        if preds[most_likely_idx] >= self.magic_threshold:
            # return the integer with highest probability in the abstract
            return abstract_tokens[numeric_token_indices[most_likely_idx]], preds[most_likely_idx]
        else:
            return None

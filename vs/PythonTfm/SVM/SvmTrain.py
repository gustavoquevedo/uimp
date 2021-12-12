from sys import prefix
from SvmHelper import Svm_helper


import pandas as pd
import numpy as np

from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer

datasets = [
    ['full', True],    
    ['small', True],
    ['imbalanced', True],
    ['oos-plus', True],
    ['undersample', False],
    ['wiki-aug', False],
]
class Print_options(Enum):
    NONE = 1
    ACCURACY = 2
    RECALL = 3
    
#def post_processing:    
    # Add the Data using pandas
    #Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\NLP Project\corpus_small.csv",encoding='latin-1')

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    # Train_X_tokenized= [word_tokenize(entry) for entry in Train_X]
    # Test_X_tokenized = [word_tokenize(entry) for entry in Test_X]

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.

    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    # tag_map = defaultdict(lambda : wn.NOUN)
    # tag_map['J'] = wn.ADJ
    # tag_map['V'] = wn.VERB
    # tag_map['R'] = wn.ADV


    # for index,entry in enumerate(Corpus['text']):
    #     # Declaring Empty List to store the words that follow the rules for this step
    #     Final_words = []
    #     # Initializing WordNetLemmatizer()
    #     word_Lemmatized = WordNetLemmatizer()
    #     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    #     for word, tag in pos_tag(entry):
    #         # Below condition is to check for Stop words and consider only alphabets
    #         if word not in stopwords.words('english') and word.isalpha():
    #             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
    #             Final_words.append(word_Final)
    #     # The final processed set of words for each iteration will be stored in 'text_final'
    #     Corpus = Corpus[~Corpus.index.duplicated()]

    #     Corpus.loc[index,'text_final'] = str(Final_words)

tfidf_max_features = 5000
def get_model_tfidf(df_train):

    #Set Random seed
    np.random.seed(500)

    df_train.columns = ['text', 'label']

    # Step - 2: Split the model into Train and Test Data set
    #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)
    train_x = df_train['text']
    train_y = df_train['label']
        
    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    label_encoder = LabelEncoder()
    train_y_encoded = label_encoder.fit_transform(np.array(train_y))

    tfidf_vect = TfidfVectorizer(max_features=tfidf_max_features)
    train_x_tfidf = tfidf_vect.fit_transform(train_x)
    
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model.fit(train_x_tfidf,train_y_encoded)
    return model, label_encoder, tfidf_vect


def get_model_onehot(df_train):
    df_train.columns = ['text', 'label']
    train_x = df_train['text']
    train_y = df_train['label']
    
    label_encoder = LabelEncoder()
    freq = CountVectorizer()
    corpus = freq.fit_transform(train_x)
    onehot = Binarizer()

    train_x_onehot = onehot.fit_transform(corpus)
    train_y_encoded = label_encoder.fit_transform(np.array(train_y))
        
    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model.fit(train_x_onehot,train_y_encoded)

    return model, label_encoder, freq, onehot

def predict_onehot(model, df_test, dataset_name, label_encoder, freq, onehot, print_options):
    df_test.columns = ['text', 'label']
    test_x = df_test['text']
    test_y = df_test['label']
    
    test_y_encoded = label_encoder.transform(np.array(test_y))
    test_x_onehot = freq.transform(test_x)
    test_x_onehot = onehot.transform(test_x_onehot)
        
    predictions_svm = model.predict(test_x_onehot)
    predictions_svm_decoded = label_encoder.inverse_transform(predictions_svm)
    
    print('\r\n' + dataset_name)
    if print_options == Print_options.ACCURACY:
      print("(OneHot) SVM Accuracy Score -> ",accuracy_score(predictions_svm, test_y_encoded)*100)
      
    if print_options == Print_options.RECALL:
      recall = recall_score(test_y, predictions_svm_decoded, average=None, labels=['oos'])
      print("(OneHot) Out-Of-Scope Recall -> ",recall*100)

    return list(zip(test_x.values, predictions_svm_decoded))

def get_model_and_predict_tfidf(df_train, df_test, ds_name, print_options):
    model, label_encoder, tfidf_vect = get_model_tfidf(df_train)        
    return predict_tfidf(model, df_test, label_encoder, ds_name, tfidf_vect, print_options)

def get_model_and_predict_onehot(df_train, df_test, ds_name, print_options):
    model, label_encoder, freq, onehot = get_model_onehot(df_train)
    return predict_onehot(model, df_test, ds_name, label_encoder, freq, onehot, print_options)

def predict_tfidf(model, df_test, label_encoder, dataset_name, tfidf_vect, print_options):    
    df_test.columns = ['text', 'label']
    test_x = df_test['text']
    test_y = df_test['label']

    test_x_tfidf = tfidf_vect.transform(test_x)
    test_y_encoded = label_encoder.transform(np.array(test_y))

    # predict the labels on validation dataset
    predictions_svm = model.predict(test_x_tfidf)
    predictions_svm_decoded = label_encoder.inverse_transform(predictions_svm)
    
    print('\r\n' + dataset_name)
    if print_options == Print_options.ACCURACY:
      print("(Tf-idf) SVM Accuracy Score -> ",accuracy_score(predictions_svm, test_y_encoded)*100)
      
    if print_options == Print_options.RECALL:
      recall = recall_score(test_y, predictions_svm_decoded, average=None, labels=['oos'])
      print("(Tf-idf) Out-Of-Scope Recall -> ",recall*100)

    return list(zip(test_x.values, predictions_svm_decoded))

def intersect(bin_predictions, df_full_test):
    bin_predictions_x = [item[0] for item in bin_predictions]
    return df_full_test[df_full_test.iloc[:, 0].isin(bin_predictions_x)]

#############

helper = Svm_helper()

df_full_train_in = helper.read_remote_json_as_dataframe('full', 'train')        
df_full_train_all = helper.read_remote_json_as_dataframe('full', 'train', True)        
df_full_test_in = helper.read_remote_json_as_dataframe('full', 'test')
df_full_test_all = helper.read_remote_json_as_dataframe('full', 'test', True)

model_full_tfidf_in, label_encoder_full_in, tfidf_vect_full_in = get_model_tfidf(df_full_train_in)
model_full_tfidf_all, label_encoder_full_all, tfidf_vect_full_all = get_model_tfidf(df_full_train_all)

model_full_onehot_in, label_encoder_full_onehot_in, freq_full_in, onehot_full_in = get_model_onehot(df_full_train_in)
model_full_onehot_all, label_encoder_full_onehot_all, freq_full_all, onehot_full_all = get_model_onehot(df_full_train_all)


for item in datasets:
    ds_name = item[0]

    df_train_in = helper.read_remote_json_as_dataframe(ds_name, 'train')
    df_val_in = helper.read_remote_json_as_dataframe(ds_name, 'val')
    df_test_in = helper.read_remote_json_as_dataframe(ds_name, 'test')
    
    non_binary = item[1]
    if non_binary: #boolean value denotes a separate file for oos
        df_train_oos = helper.read_remote_json_as_dataframe(ds_name, 'train', True)
        df_train = pd.concat([df_train_in, df_train_oos])
        
        df_val_oos = helper.read_remote_json_as_dataframe(ds_name, 'val', True)
        df_val = pd.concat([df_val_in, df_val_oos])
        
        df_test_oos = helper.read_remote_json_as_dataframe(ds_name, 'test', True)
        df_test = pd.concat([df_test_in, df_test_oos])        

        get_model_and_predict_tfidf(df_train_in, df_test_in, ds_name, Print_options.ACCURACY)
        get_model_and_predict_tfidf(df_train, df_test, ds_name, Print_options.RECALL)
        
        get_model_and_predict_onehot(df_train_in, df_test_in, ds_name, Print_options.ACCURACY)
        get_model_and_predict_onehot(df_train, df_test, ds_name, Print_options.RECALL)

    else:
        df_train_all = df_train_in
        df_val_all = df_val_in
        df_test_all = df_test_in
                
        ## tf-idf    
        bin_predictions_tfidf = get_model_and_predict_tfidf(df_train_all, df_test_all, ds_name, Print_options.NONE)      
        
        bin_predictions_tfidf_in = list(filter(lambda x : x[1] == 'in', bin_predictions_tfidf))
        full_predictions_tfidf_in = intersect(bin_predictions_tfidf_in, df_full_test_in)
        predict_tfidf(model_full_tfidf_in, full_predictions_tfidf_in, label_encoder_full_in, ds_name, tfidf_vect_full_in, Print_options.ACCURACY)
        
        full_predictions_tfidf = intersect(bin_predictions_tfidf, df_full_test_all)
        predict_tfidf(model_full_tfidf_all, full_predictions_tfidf, label_encoder_full_all, ds_name, tfidf_vect_full_all, Print_options.RECALL)
        
        ## one hot        
        bin_predictions_onehot = get_model_and_predict_onehot(df_train_all, df_test_all, ds_name, Print_options.NONE)     
        
        bin_predictions_onehot_in = list(filter(lambda x : x[1] == 'in', bin_predictions_onehot))
        full_predictions_onehot_in = intersect(bin_predictions_onehot_in, df_full_test_in) 
        predict_onehot(model_full_onehot_in, full_predictions_onehot_in, ds_name, label_encoder_full_onehot_in, freq_full_in, onehot_full_in, Print_options.ACCURACY)
        
        full_predictions_onehot = intersect(bin_predictions_onehot, df_full_test_all)
        predict_onehot(model_full_onehot_all, full_predictions_onehot, ds_name, label_encoder_full_onehot_all, freq_full_all, onehot_full_all, Print_options.RECALL)
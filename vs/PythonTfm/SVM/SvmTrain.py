from SvmHelper import Svm_helper


import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

structure = [
    ['full', True],    
    ['small', True],
    ['imbalanced', True],
    ['oos-plus', True],
    ['undersample', False],
    ['wiki-aug', False],
]

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

def run(dataset_name, df_train, df_test, print_recall):

    #Set Random seed
    np.random.seed(500)

    df_train.columns = ['text', 'label']
    df_test.columns = ['text', 'label']

    # Step - 2: Split the model into Train and Test Data set
    #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)
    train_x = df_train['text']
    test_x = df_test['text']
    train_y = df_train['label']
    test_y = df_test['label']
        
    # Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
    label_encoder = LabelEncoder()
    train_y_encoded = label_encoder.fit_transform(np.array(train_y))
    test_y_encoded = label_encoder.fit_transform(np.array(test_y))

    # Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(np.array(train_x))

    train_x_tfidf = tfidf_vect.transform(train_x)
    test_x_tfidf = tfidf_vect.transform(test_x)
    
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    model.fit(train_x_tfidf,train_y_encoded)

    # predict the labels on validation dataset
    predictions_svm = model.predict(test_x_tfidf)
    predictions_svm_decoded = label_encoder.inverse_transform(predictions_svm)

    # Use accuracy_score function to get the accuracy
    print('\r\n' + dataset_name)
    if print_recall:
      recall = recall_score(Test_Y, predictions_svm_decoded, average=None, labels=['oos'])
      print("Out-Of-Scope Recall -> ",recall*100)
    else:
      print("SVM Accuracy Score -> ",accuracy_score(predictions_svm, test_y_encoded)*100)


from urllib.request import urlopen
import json

helper = Svm_helper()

for item in structure:
    df_train_in = helper.read_remote_json_as_dataframe(item[0], 'train')
    df_val_in = helper.read_remote_json_as_dataframe(item[0], 'val')
    df_test_in = helper.read_remote_json_as_dataframe(item[0], 'test')

    if item[1]: #boolean value denotes a separate file for oos
        df_train_oos = helper.read_remote_json_as_dataframe(item[0], 'train', True)
        df_train = pd.concat([df_train_in, df_train_oos])
        
        df_val_oos = helper.read_remote_json_as_dataframe(item[0], 'val', True)
        df_val = pd.concat([df_val_in, df_val_oos])
        
        df_test_oos = helper.read_remote_json_as_dataframe(item[0], 'test', True)
        df_test = pd.concat([df_test_in, df_test_oos])

    df_train_val_in = pd.concat([df_train_in, df_val_in])
    df_train_val = pd.concat([df_train, df_val])

    run(item[0] + ", train only", df_train_in, df_test_in, False)
    run(item[0] + ", train + val", df_train_val_in, df_test_in, False)
    run(item[0] + ", train only", df_train, df_test, True)
    run(item[0] + ", train + val", df_train_val, df_test, True)
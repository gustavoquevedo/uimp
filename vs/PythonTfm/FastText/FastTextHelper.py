import fasttext
from urllib.request import urlopen
import json
import os.path
from array import *

from numpy.core.numeric import True_
import CommonHelper
from CommonHelper import *
import spacy

        
nlp = spacy.load("en_core_web_lg")
exec_params = ExecParams.get_default_exec_params()

class DatasetHyperparameters:
    def __init__(self, title, lr, dim, wordNgrams, lrUpdateRate, ws, loss, n_epochs = 100, min_epochs = 40, max_epochs = 140):
        self.title = title
        self.lr = lr
        self.dim = dim
        self.wordNgrams = wordNgrams
        self.lrUpdateRate = lrUpdateRate
        self.ws = ws
        self.loss = loss
        self.n_epochs = n_epochs
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs


#def create_file(data_json: list, fasttext_filepath: str):
#    with open(fasttext_filepath, 'w') as f:
#        for record in data_json:
#            f.write('__label__' + record[1] + ' ' + record[0])
#            f.write('\n')

###########
## model ##
###########

def create_model(hyp:DatasetHyperparameters, train_path):
    model = fasttext.train_supervised(
        input=train_path, 
        epoch=hyp.n_epochs, 
        lr=hyp.lr, 
        dim=hyp.dim, 
        wordNgrams=hyp.wordNgrams, 
        lrUpdateRate=hyp.lrUpdateRate, 
        ws=hyp.ws, 
        loss=hyp.loss)
    return model
    
def is_in_scope_predicted(model, text):
    return model.predict(text)[0][0] == '__label__in'

def create_and_evaluate_model(ds_name, hyperparam):
    train_filepath = helper.get_dataset_local_path(ds_name, 'train')
    print("\r\n" + "Creating model " + hyperparam.title + " from file " + train_filepath + "...\r\n")
    model = create_model(hyperparam, train_filepath)
    
    test_filepath = helper.get_dataset_local_path(ds_name, 'test')
    print(model.test(test_filepath))
    test_label_response = model.test_label(test_filepath)
    print(test_label_response['__label__oos'])

    
def run_in_scope_accuracy(hyperp, train_filepath, test_filepath):
    results = []
    for i in range(hyperp.min_epochs, hyperp.max_epochs + 1):
        if i % 5 == 0:
            print('\r\nEpochs: ' + str(i))
            hyperp.epochs = i
            model = create_model(hyperp, train_filepath)
            result = model.test(test_filepath)
            results.append([i, result[1]])
            print(result)
    print(results)
            
def run_oos_recall(hyperp, train_filepath, test_filepath): 
    for i in range(hyperp.min_epochs, hyperp.max_epochs + 1):
        if i % 5 == 0:
            print('\r\nEpochs: ' + str(i))
            hyperp.epochs = i
            model = create_model(hyperp, train_filepath)
            test_label_response = model.test_label(test_filepath)
            if '__label__oos' in test_label_response:
                print(test_label_response['__label__oos'])
    
#############
## content ##
#############


def get_fasttext_line(record, fasttext_content_mode, spacy_tags_mode):
    original_sentence = record[0]
    line = '__label__' + record[1] + ' ' 

    if exec_params.content_mode == ContentMode.no_filter:
        return line + original_sentence
    else:
        doc = nlp(record[0])
        if exec_params.spacy_tags_mode == SpacyTagsMode.pos_tags:
            relevant_words = [t.text for t in doc if t.pos_ in exec_params.relevant_tags]
        elif exec_params.spacy_tags_mode == SpacyTagsMode.dep_tags:
            relevant_words = [t.text for t in doc if t.dep_ in exec_params.relevant_tags]

        if exec_params.content_mode == ContentMode.duplicate_relevant:
            line += original_sentence + ' ' + ' '.join(relevant_words)
        elif exec_params.content_mode == ContentMode.remove_non_relevant:
            line += ' ' + ' '.join(relevant_words)

    return line

###############
# write files #
###############

def get_dataset_local_path(dataset_name, partition_name, folder = "", scope = ScopeSet.all):
    current_path = os.path.abspath(os.path.dirname("__file__"))
    if folder != "":
        current_path += "\\" + folder
    if scope != ScopeSet.all:
        current_path += "_" + ScopeSet.in_.name

    fasttext_filename = get_fasttext_filename(dataset_name, partition_name)
    return os.path.join(current_path, fasttext_filename)

def get_fasttext_filename(dataset_name, partition_name):
    return 'fasttext_' + dataset_name + '.' + partition_name

def write_fasttext_records(ds_name, partition_name, separate_file_for_oos = False):        
    helper = CommonHelper.CommonHelper()
    with open(get_fasttext_filename(ds_name, partition_name), 'w') as f:
        data_json = helper.get_json_data(ds_name, partition_name)
        data_json_oos = None
        for record in data_json:
            line = get_fasttext_line(record, exec_params.content_mode, exec_params.spacy_tags_mode)
            f.write(line + '\n')
          
        if separate_file_for_oos: 
            data_json_oos = helper.get_json_data_oos(ds_name, partition_name)
            for record_oos in data_json_oos:
                line = get_fasttext_line(record_oos, exec_params.content_mode, exec_params.spacy_tags_mode)
                f.write(line + '\n')

def write_fasttext_records_separate_oos(ds_name, partition_name):
    return write_fasttext_records(ds_name, partition_name, True)

#def create_model(hyp:DatasetHyperparameters, train_path):
#    print("\r\n" + "Creating model " + hyp.title + "...\r\n")
#    model = fasttext.train_supervised(
#        input=train_path, 
#        epoch=hyp.n_epochs, 
#        lr=hyp.lr, 
#        dim=hyp.dim, 
#        wordNgrams=hyp.wordNgrams, 
#        lrUpdateRate=hyp.lrUpdateRate, 
#        ws=hyp.ws, 
#        loss=hyp.loss)
#    return model
    
#def get_dataset_local_path(dataset_name, partition_name):
#    current_path = os.path.abspath(os.path.dirname(__file__))    
#    fasttext_filename = 'fasttext_' + dataset_name + '.' + partition_name
#    return os.path.join(current_path, fasttext_filename)

#def get_dataset_remote_path(dataset_name, partition_name, oos_extra_file = False):
#    if oos_extra_file:
#        filename = dataset_name + '/oos_' + partition_name + '.json'
#    else:
#        filename = dataset_name + '/' + partition_name + '.json'
#    return 'https://raw.githubusercontent.com/gustavoquevedo/uimp/master/oos-eval/' + filename

    
#def read_remote_json(dataset_name, partition_name, has_oos_extra_file = False):
#    url = get_dataset_remote_path(dataset_name, partition_name)
#    response = urlopen(url)
#    data_json = json.loads(response.read())

#    if has_oos_extra_file:
#        url_extra =get_dataset_remote_path(dataset_name, partition_name, True)
#        response = urlopen(url_extra)
#        data_json_extra = json.loads(response.read())
#        data_json += data_json_extra

#    return data_json

#def create_file(data_json: list, fasttext_filepath: str):
#    with open(fasttext_filepath, 'w') as f:
#        for record in data_json:
#            f.write('__label__' + record[1] + ' ' + record[0])
#            f.write('\n')

#def create_file_from_remote_json(dataset_name, partition_name, have_oos_extra_file = False):
#    json = read_remote_json(dataset_name, partition_name, have_oos_extra_file)
#    filepath = get_dataset_local_path(dataset_name, partition_name)
#    create_file(json, filepath)
#    return filepath, json
    
#def is_text_in_dataset(text, data_json):
#    for item in data_json:
#        if item[0] == text:
#            return True
#    return False

#def get_label_or_oos(text, data_json):
#    item = next((x[1] for x in data_json if x[0] == text), None)
#    return item if item is not None else 'oos'

#def is_in_scope_predicted(model, text):
#    return model.predict(text)[0][0] == '__label__in'
    
#def create_full_model(include_oos):    
#    # create files and model
#    full_train_filepath, full_train_json = create_file_from_remote_json('full', 'train', include_oos)
#    return create_model(get_full_hyperparam(), full_train_filepath)
    
#def run_in_scope_accuracy(hyperp, test_partition_name = 'test'):    
#    train_filepath, train_data_json = create_file_from_remote_json(hyperp.title, 'train')
#    test_filepath, test_data_json = create_file_from_remote_json(hyperp.title, test_partition_name)
#    for i in range(hyperp.min_epochs, hyperp.max_epochs):
#        if i % 5 == 0:
#            print('\r\nEpochs: ' + str(i))
#            hyperp.epochs = i
#            model = create_model(hyperp, train_filepath)
#            print(model.test(test_filepath))

#def run_oos_recall(hyperp, test_partition_name = 'test'):
#    train_filepath, train_data_json = create_file_from_remote_json(hyperp.title, 'train', True)
#    test_filepath, test_data_json = create_file_from_remote_json(hyperp.title, test_partition_name, True)        
#    for i in range(hyperp.min_epochs, hyperp.max_epochs):
#        if i % 5 == 0:
#            print('\r\nEpochs: ' + str(i))
#            hyperp.epochs = i
#            model = create_model(hyperp, train_filepath)
#            test_label_response = model.test_label(test_filepath)
#            if '__label__oos' in test_label_response:
#                print(test_label_response['__label__oos'])
    
#def run_binary(hyperp, full_model_in, full_model_all, test_in, test_all, test_partition_name = 'test'):
#    for i in range(hyperp.min_epochs, hyperp.max_epochs):
#        if i % 5 == 0:
#            print('\r\nEpochs: ' + str(i))
#            hyperp.n_epochs = i
#            # create files and model
#            train_filepath, train_json  = create_file_from_remote_json(hyperp.title, 'train')
#            test_filepath, test_json  = create_file_from_remote_json(hyperp.title, test_partition_name)
#            model = create_model(hyperp, train_filepath)

#            test_label_response = model.test_label(test_filepath)
#            if '__label__oos' in test_label_response:
#                print(test_label_response['__label__oos'])
    
#            # get json list (in scope only) from test partition 
#            test_json_predicted_in = list(filter(lambda x : is_in_scope_predicted(model, x[0]), test_in))
#            # create file with items in list above and test
#            create_file(test_json_predicted_in, test_filepath)
#            result = full_model_in.test(test_filepath)
#            print(result)
    
#            ## get json list (in scope only) from test partition 
#            #test_json_predicted_all = list(filter(lambda x : is_in_scope_predicted(model, x[0]), test_all))
#            ## create file with items in list above and test
#            #create_file(test_json_predicted_all, test_filepath)
#            #test_label_response = full_model_all.test_label(test_filepath)
#            #if '__label__oos' in test_label_response:
#            #    print(test_label_response['__label__oos'])


def get_full_hyperparam():
    return DatasetHyperparameters(
        title='full',
        lr=1.0,
        dim=100,
        wordNgrams=3,
        lrUpdateRate=100,
        ws=3,
        loss="ns",
        min_epochs=50,
        max_epochs=180)
        
def get_undersample_hyperparam():
    return DatasetHyperparameters(
        title='undersample',
        lr=0.1,
        dim=200,
        wordNgrams=4,
        lrUpdateRate=100,
        ws=3,
        loss='ns',
        min_epochs=30,
        max_epochs=130)

def get_wiki_aug_hyperparam():
    return DatasetHyperparameters(
        title='wiki-aug',
        lr=1,
        dim=400,
        wordNgrams=4,
        lrUpdateRate=100,
        ws=3,
        loss='ns',
        min_epochs=80,
        max_epochs=240)
   
def get_small_hyperparam():
    return DatasetHyperparameters(
        title='small',
        lr=0.1,
        dim=200,
        wordNgrams=1,
        lrUpdateRate=200,
        ws=7,
        loss='softmax',
        min_epochs=120,
        max_epochs=220)
   
def get_imbalanced_hyperparam():
    return DatasetHyperparameters(
        title='imbalanced',
        lr=1,
        dim=400,
        wordNgrams=2,
        lrUpdateRate=200,
        ws=7,
        loss='ns',
        min_epochs=50,
        max_epochs=140)   
   
def get_oos_plus_hyperparam():
    return DatasetHyperparameters(
        title='oos-plus',
        lr=1,
        dim=50,
        wordNgrams=2,
        lrUpdateRate=200,
        ws=7,
        loss='ns',
        min_epochs=40,
        max_epochs=140)


from FastTextHelper import Fasttext_helper

#!pip install fasttext==0.9.1
#!pip install fasttext # would install 0.9.2 with an issue so that nan always is returned for label recall

helper = Fasttext_helper()

structure_train = [
    helper.get_full_hyperparam(),
    helper.get_imbalanced_hyperparam(),
    helper.get_oos_plus_hyperparam(),
    helper.get_small_hyperparam()
]

structure_binary = [
    helper.get_undersample_hyperparam(),
    helper.get_wiki_aug_hyperparam()
]


##########

for hyperp in structure_train:
    print('\r\n' + hyperp.title + '\r\n===')     
    helper.run_in_scope_accuracy(hyperp, 'val')
    helper.run_oos_recall(hyperp, 'val')
    
full_model_in = helper.create_full_model(include_oos=False)
full_model_all = helper.create_full_model(include_oos=True)
test_json_in = helper.read_remote_json('full', 'val', has_oos_extra_file=False)
test_json_all = helper.read_remote_json('full', 'val', has_oos_extra_file=True)

for hyperp in structure_binary:
    print('\r\n' + hyperp.title + '\r\n===')         
    helper.run_binary(hyperp, full_model_in, full_model_all, test_json_in, test_json_all, 'test')


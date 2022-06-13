from CommonHelper import *
from FastTextHelper import *

structure_train = [
    #get_full_hyperparam(),
    #get_imbalanced_hyperparam(),
    #get_oos_plus_hyperparam(),
    get_small_hyperparam()
]

structure_binary = [
    get_undersample_hyperparam(),
    #get_wiki_aug_hyperparam()
]

structure_variations = [
    # 'dup_dep_v1',
    # 'dup_pos_v1',
    'dup_pos_v2',
    # 'rem_dep_v1',
    # 'rem_pos_v2'
]

##########

for variation in structure_variations:
    test_file_path_in = get_dataset_local_path('full', 'test', variation, ScopeSet.in_)
    test_file_path_all = get_dataset_local_path('full', 'test', variation, ScopeSet.all)
    
    print('\r\n' + variation + '\r\n===')

    for hyperp in structure_train:        
        train_file_path_in = get_dataset_local_path(hyperp.title, 'train', variation, ScopeSet.in_)
        train_file_path_all = get_dataset_local_path(hyperp.title, 'train', variation, ScopeSet.all)
        
        print('\r\n' + hyperp.title + '\r\n===')
        print('\r\n' + 'In scope accuracy' + '\r\n')
        run_in_scope_accuracy(hyperp, train_file_path_in, test_file_path_in)
        print('\r\n' + 'Out of scope recall' + '\r\n')
        run_oos_recall(hyperp, train_file_path_all, test_file_path_all)
    
#full_model_in = create_full_model(include_oos=False)
#full_model_all = create_full_model(include_oos=True)
#test_json_in = read_remote_json('full', 'test', has_oos_extra_file=False)
#test_json_all = read_remote_json('full', 'test', has_oos_extra_file=True)

#for hyperp in structure_binary:
#    print('\r\n' + hyperp.title + '\r\n===')         
#    run_binary(hyperp, full_model_in, full_model_all, test_json_in, test_json_all, 'test')


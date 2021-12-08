from typing_extensions import IntVar
import fasttext
from urllib.request import urlopen
import json
import os.path
import pandas as pd

class Svm_helper:    
    def get_dataset_remote_path(self, dataset_name, partition_name, oos_extra_file = False):
        if oos_extra_file:
            filename = dataset_name + '/oos_' + partition_name + '.json'
        else:
            filename = dataset_name + '/' + partition_name + '.json'
        return 'https://raw.githubusercontent.com/gustavoquevedo/uimp/master/oos-eval/' + filename
        
    def read_remote_json_as_dataframe(self, dataset_name, partition_name, has_oos_extra_file = False):
        url = self.get_dataset_remote_path(dataset_name, partition_name)
        response = urlopen(url)
        data_json = json.loads(response.read())

        if has_oos_extra_file:
            url_extra =self.get_dataset_remote_path(dataset_name, partition_name, True)
            response = urlopen(url_extra)
            data_json_extra = json.loads(response.read())
            data_json += data_json_extra

        return pd.DataFrame(data_json)
        
    def is_text_in_dataset(self, text, data_json):
        for item in data_json:
            if item[0] == text:
                return True
        return False

    def get_label_or_oos(self, text, data_json):
        item = next((x[1] for x in data_json if x[0] == text), None)
        return item if item is not None else 'oos'
        
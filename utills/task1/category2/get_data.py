import os
import json
def get_data(config):
    
    train_dir = config.train_link
    valid_dir = config.valid_link
    train_dir_list = os.listdir(train_dir)
    valid_dir_list = os.listdir(valid_dir)
    train_sample_list = []
    valid_sample_list = []

    for directory in train_dir_list:
        FOLDER_NAME = directory+"/"
        JSON_NAME = directory+".json"
        FILE_PATH = train_dir+FOLDER_NAME
        a_data = json.load(open(FILE_PATH+JSON_NAME))
        a_data["file_path"] = FILE_PATH
        a_data["question"] = [a_data["Questions"][0]]
        a_data["answer"] = [a_data["Answers"][0]]
        del a_data["Answers"]
        del a_data["Questions"]
        train_sample_list.append(a_data)

    for directory in valid_dir_list:
        FOLDER_NAME = directory+"/"
        JSON_NAME = directory+".json"
        FILE_PATH = valid_dir+FOLDER_NAME
        a_data = json.load(open(FILE_PATH+JSON_NAME))
        a_data["file_path"] = FILE_PATH
        a_data["question"] = [a_data["Questions"][0]]
        a_data["answer"] = [a_data["Answers"][0]]
        del a_data["Answers"]
        del a_data["Questions"]
        valid_sample_list.append(a_data)

        
    return train_sample_list, valid_sample_list
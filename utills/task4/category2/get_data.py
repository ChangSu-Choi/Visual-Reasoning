import os
import json
def get_data(config):
    
    train_dir = config.train_link
    valid_dir = config.valid_link

    train_sample_list = []
    valid_sample_list = []
    for root, dirs, files in os.walk(train_dir):
        if root == train_dir:  # only process first level directories
            for directory_parent in dirs:
                FOLDER_NAME = directory_parent+"/"
                JSON_NAME = directory_parent+".json"
                FILE_PATH = os.path.join(root, FOLDER_NAME)
                train_dir_list = os.listdir(FILE_PATH)

                for directory in train_dir_list:
                    JSON_NAME = directory+"/"+directory+".json"
                    a_data = json.load(open(FILE_PATH+JSON_NAME))
                    a_data["file_path"] = FILE_PATH+directory+"/"
                    a_data["answer_img1"] = a_data["Answers"][0]
                    a_data["answer_img2"] = a_data["Answers"][1]
                    a_data["answer_img3"] = a_data["Answers"][2]
                    a_data["answer_img4"] = a_data["Answers"][3]
                    a_data["answer_img5"] = a_data["Answers"][4]

                    del a_data["Answers"]


                    train_sample_list.append(a_data)

    for root, dirs, files in os.walk(valid_dir):
        if root == valid_dir:  # only process first level directories
            for directory_parent in dirs:
                FOLDER_NAME = directory_parent+"/"
                JSON_NAME = directory_parent+".json"
                FILE_PATH = os.path.join(root, FOLDER_NAME)
                valid_dir_list = os.listdir(FILE_PATH)


                for directory in valid_dir_list:
                    JSON_NAME = directory+"/"+directory+".json"
                    a_data = json.load(open(FILE_PATH+JSON_NAME))
                    a_data["file_path"] = FILE_PATH+directory+"/"
                    a_data["answer_img1"] = a_data["Answers"][0]
                    a_data["answer_img2"] = a_data["Answers"][1]
                    a_data["answer_img3"] = a_data["Answers"][2]
                    a_data["answer_img4"] = a_data["Answers"][3]
                    a_data["answer_img5"] = a_data["Answers"][4]
                    del a_data["Answers"]
                    valid_sample_list.append(a_data)

    return train_sample_list, valid_sample_list
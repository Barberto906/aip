import os
import shutil

if __name__ == "__main__":

    abs_path_to_current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_market = os.path.join(abs_path_to_current_dir, 'market1501')
    path_to_cuhk = os.path.join(abs_path_to_current_dir, 'cuhk03-np')

    path_to_trainset_market = os.path.join(path_to_market, 'bounding_box_train')
    path_to_testset_market = os.path.join(path_to_market, 'bounding_box_test')
    path_to_queryset_market = os.path.join(path_to_market, 'query')

    path_to_trainset_cuhk = os.path.join(path_to_cuhk, 'detected/bounding_box_train')
    path_to_testset_cuhk = os.path.join(path_to_cuhk, 'detected/bounding_box_test')
    path_to_queryset_cuhk = os.path.join(path_to_cuhk, 'detected/query')

    path_to_final_trainset = os.path.join(abs_path_to_current_dir, 'FinalDataset\\bounding_box_train')
    path_to_final_testset = os.path.join(abs_path_to_current_dir, 'FinalDataset\\bounding_box_test')
    path_to_final_queryset = os.path.join(abs_path_to_current_dir, 'FinalDataset\\query')

    os.mkdir('FinalDataset')
    os.mkdir(path_to_final_trainset)
    os.mkdir(path_to_final_testset)
    os.mkdir(path_to_final_queryset)

    # COPYING MARKET DATASET:
    # TRAINSET
    for file_name in os.listdir(path_to_trainset_market):
        if 'jpg' in file_name:
            path_to_file = os.path.join(path_to_trainset_market, file_name)
            shutil.copy2(path_to_file, path_to_final_trainset)
    # TESTSET
    for file_name in os.listdir(path_to_testset_market):
        if 'jpg' in file_name:
            if file_name[0:2] == "-1":
                continue
            if file_name[0:4] == "0000":
                continue
            path_to_file = os.path.join(path_to_testset_market, file_name)
            shutil.copy2(path_to_file, path_to_final_testset)
    # QUERYSET
    for file_name in os.listdir(path_to_queryset_market):
        if 'jpg' in file_name:
            path_to_file = os.path.join(path_to_queryset_market, file_name)
            shutil.copy2(path_to_file, path_to_final_queryset)

# COPYING CUHK DATASET:
    # TRAINSET
    for file_name in os.listdir(path_to_trainset_cuhk):
        if 'png' in file_name:
            path_to_file = os.path.join(path_to_trainset_cuhk, file_name)
            new_id = int(file_name[0:4])+1600
            new_file_name = str(new_id)+file_name[4:]
            path_to_final_filename = os.path.join(path_to_final_trainset, new_file_name)
            shutil.copy2(path_to_file, path_to_final_filename)
    # TESTSET
    for file_name in os.listdir(path_to_testset_cuhk):
        if 'png' in file_name:
            path_to_file = os.path.join(path_to_testset_cuhk, file_name)
            new_id = int(file_name[0:4]) + 1600
            new_file_name = str(new_id) + file_name[4:]
            path_to_final_filename = os.path.join(path_to_final_testset, new_file_name)
            shutil.copy2(path_to_file, path_to_final_filename)
    # QUERYSET

    for file_name in os.listdir(path_to_queryset_cuhk):
        if 'png' in file_name:
            path_to_file = os.path.join(path_to_queryset_cuhk, file_name)
            new_id = int(file_name[0:4]) + 1600
            new_file_name = str(new_id) + file_name[4:]
            path_to_final_filename = os.path.join(path_to_final_queryset, new_file_name)
            shutil.copy2(path_to_file, path_to_final_filename)
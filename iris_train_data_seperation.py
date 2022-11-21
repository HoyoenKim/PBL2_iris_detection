import json
import os
import shutil

with open("./env_var.json", "r") as env_var_json:
    env_var = json.load(env_var_json)
eye_data_path = env_var['eye_data_path']
image_dir_path = env_var['train_image_dir_path']
mask_dir_path = env_var['train_mask_dir_path']

if __name__ == "__main__":
    if os.path.exists(eye_data_path):
        eye_data_list = os.listdir(eye_data_path)
        if not os.path.exists(image_dir_path):
            print('----------------------------------')
            print('There is no IMG dir, ... generate.')
            print('----------------------------------')
            os.mkdir(image_dir_path)

        if not os.path.exists(mask_dir_path):
            print('-----------------------------------')
            print('There is no MASK dir, ... generate.')
            print('-----------------------------------')
            os.mkdir(mask_dir_path)

        print('Copying...')
        for eye_data in eye_data_list:
            filename = os.path.basename(eye_data)
            ext = os.path.splitext(eye_data)[1]
            if ext == '.jpg':
                #print(os.path.join('./single_eye', d_path), os.path.join('./PNG', filename))
                shutil.copy(os.path.join(eye_data_path, eye_data), os.path.join(image_dir_path, filename))
            elif ext == '.png':
                #print(os.path.join('./single_eye', d_path), os.path.join('./Mask', filename))
                shutil.copy(os.path.join(eye_data_path, eye_data), os.path.join(mask_dir_path, filename))
    else:
        print('---------------------')
        print('There is no eye data.')
        print('---------------------')
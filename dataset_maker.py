import argparse
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import h5py
import cv2
from typing import List, Union

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_out_path', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/validation_fold_5.csv', help="Directory to save paths to all chest x-ray images in dataset.")
    parser.add_argument('--cxr_out_path', type=str, default='/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/validation_fold_5.h5', help="Directory to save processed chest x-ray image data.")
    parser.add_argument('--dataset_type', type=str, default='mimic', choices=['mimic', 'chexpert-test'], help="Type of dataset to pre-process")
    parser.add_argument('--chest_x_ray_path', default='/home/woody/iwi5/iwi5207h/dataset/mimic-cxr-jpg/2.1.0', help="Directory where chest x-ray image data is stored. This should point to the files folder from the MIMIC chest x-ray dataset.")
    parser.add_argument('--csv_input_path', default='/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/valid_fold_5.csv', help="CSV file containing the image paths to process.")
    args = parser.parse_args()
    return args

def load_data(filepath):
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_cxr_paths_list(csv_input_path, chest_x_ray_path): 
    # Load the CSV file containing image paths
    dataframe = load_data(csv_input_path)
    
    # Assume the CSV has a column named 'Path' which contains relative paths of the images
    cxr_paths = dataframe['fpath'].apply(lambda x: os.path.join(chest_x_ray_path, x))
    
    return cxr_paths

'''
This function resizes and zero pads image 
'''
def preprocess(img, desired_size=320):
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    # create a new image and paste the resized on it

    new_img = Image.new('L', (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

def img_to_hdf5(cxr_paths: List[Union[str, Path]], out_filepath: str, resolution=320): 
    """
    Convert directory of images into a .h5 file given paths to all 
    images. 
    """
    dset_size = len(cxr_paths)
    failed_images = []
    with h5py.File(out_filepath,'w') as h5f:
        img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution))    
        for idx, path in enumerate(tqdm(cxr_paths)):
            try: 
                # read image using cv2
                img = cv2.imread(str(path))
                # convert to PIL Image object
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                # preprocess
                img = preprocess(img_pil, desired_size=resolution)     
                img_dset[idx] = img
            except Exception as e: 
                failed_images.append((path, e))
    print(f"{len(failed_images)} / {len(cxr_paths)} images failed to be added to h5.", failed_images)

if __name__ == "__main__":
    args = parse_args()
    
    # Load paths from the CSV input file
    cxr_paths = get_cxr_paths_list(args.csv_input_path, args.chest_x_ray_path)
    
    # Process images and save to HDF5
    img_to_hdf5(cxr_paths, args.cxr_out_path)

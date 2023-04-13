import tensorflow
import tensorflow as tf
import itertools
import numpy as np
from editdistance import eval as edit_distance
from tqdm import tqdm
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import cv2
import random
import itertools, os, time
import config
import matplotlib.pyplot as plt
from tacobox import Taco

# Make a dcit of language name & model path
data_models = {"kannada": "...add your path ../weights-kannada/Best_EASTER2--130--0.68.hdf5",
               "gujarati": "...add your path ../weights-gujarati/BEST1_EASTER2--40--1.17.hdf5",
               "malayalam": "...add your path ../weights-malaylam/BEST_EASTER2--28--0.95.hdf5",
               "tamil": "...add your path ../weights-tamil/EASTER2--100--0.81.hdf5",
               "telugu": "...add your path ../weights-telegu/best_telegu.hdf5",
               "devanagari": "...add your path ../weights-hindi/weights/EASTER2--148--1.90.hdf5"}

mytaco = Taco(
    cp_vertical=0.2,
    cp_horizontal=0.25,
    max_tw_vertical=100,
    min_tw_vertical=10,
    max_tw_horizontal
    =50,
    min_tw_horizontal=10
)

def preprocess(img, augment=True):
    if augment:
        img = apply_taco_augmentations(img)

    # scaling image [0, 1]
    img = img/255
    img = img.swapaxes(-2,-1)[...,::-1]
    target = np.ones((config.INPUT_WIDTH, config.INPUT_HEIGHT))
    new_x = config.INPUT_WIDTH/img.shape[0]
    new_y = config.INPUT_HEIGHT/img.shape[1]
    min_xy = min(new_x, new_y)
    new_x = int(img.shape[0]*min_xy)
    new_y = int(img.shape[1]*min_xy)
    img2 = cv2.resize(img, (new_y,new_x))
    target[:new_x,:new_y] = img2
    return 1 - (target)
    
def apply_taco_augmentations(input_img):
    random_value = random.random()
    if random_value <= config.TACO_AUGMENTAION_FRACTION:
        augmented_img = mytaco.apply_vertical_taco(
            input_img, 
            corruption_type='random'
        )
    else:
        augmented_img = input_img
    return augmented_img

def getTestImage():
    batchRange = range(0, len(samples))
    imgs = []
    texts = []
    reals = []
    for i in batchRange:
        img1 = cv2.imread(samples[i].filePath, cv2.IMREAD_GRAYSCALE)
        real = cv2.imread(samples[i].filePath)
        if img1 is None:
            img1 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
        img = preprocess(img1, augment=False)
        img = np.expand_dims(img,  0)
        text = samples[i].gtText
        imgs.append(img)
        texts.append(text)
        reals.append(real)
    currIdx += batchSize
    return imgs,texts,reals

def load_easter_model(checkpoint_path):
    if checkpoint_path == "Empty":
        checkpoint_path = "best_telegu.hdf5"
    try:
        checkpoint = tensorflow.keras.models.load_model(
            checkpoint_path,
            custom_objects={'<lambda>': lambda x, y: y,
            'tf':tf}
        )
        
        EASTER = tensorflow.keras.models.Model(
            checkpoint.get_layer('the_input').input,
            checkpoint.get_layer('Final').output
        )
    except:
        print ("Unable to Load Checkpoint.")
        return None
    return EASTER

def decoder(output,letters):
    ret = []
    for j in range(output.shape[0]):
        out_best = list(np.argmax(output[j,:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

def pred_single(img_path,model):
    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(img_path)
    if img1 is None:
        img1 = np.zeros([config.INPUT_WIDTH, config.INPUT_HEIGHT])
    img = preprocess(img1, augment=False)
    img = np.expand_dims(img,  0)
    output = model.predict(img)
    prediction = decoder(output, charlist)
    output = (prediction[0].strip(" ").replace("  ", " "))
    return output

def get_txt_file(lang):
    base_path_train = f'/data/BADRI/IHTR/trainset/{lang}/images/'
    base_path_test = f'/data/BADRI/IHTR/testset_small/{lang}/images/'
    base_path_val = f'/data/BADRI/IHTR/validationset_small/{lang}/images/'

    # Define file paths
    path_train = f"/data/BADRI/IHTR/trainset/{lang}/train.txt"
    path_test = f"/data/BADRI/IHTR/testset_small/{lang}/test.txt"
    path_val = f"/data/BADRI/IHTR/validationset_small/{lang}/val.txt"

    with open(path_test, 'r') as f:
        lines = f.readlines()

    concatenated_lines = []
    for line in lines:
        line = line.replace("test/","")
        img_path = base_path_test + line
        concatenated_lines.append(img_path)
    
    return concatenated_lines

# ------------------------------------------------>

from tqdm import tqdm
import json

def get_test_results(best_model_path,lang):
    model = load_easter_model(best_model_path)

    print('\n' + 50*'*' + ' : ' +lang + '\n')

    concated_lines = get_txt_file(lang)
    print(len(concated_lines))
    dict_test = {}
    for c_line in tqdm(concated_lines):
        c_line_new = c_line.replace("\n","")
        out = pred_single(c_line_new,model)
        out = out.strip()
        dict_test[(c_line_new.split("/"))[-1]] = out
        with open(f'./results/result_small_test_{lang}_.txt', 'a') as fp:
            fp.write((c_line_new.split("/"))[-1] + ":" + out + '\n')

    del model

    with open(f'./results/result_small_test_{lang}_.json', 'w') as fp:
        json.dump(dict_test, fp)

from easter_model import train


for lang in data_models:
    charlist = train(lang)
    get_test_results(data_models[lang],lang)
    
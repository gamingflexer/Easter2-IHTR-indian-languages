import config
import tensorflow
import tensorflow as tf
import itertools
import numpy as np
from editdistance import eval as edit_distance
from tqdm import tqdm
from data_loader import data_loader
import tensorflow.keras.backend as K
from gensim.utils import tokenize
import pandas as pd

results = []

def ctc_custom(args):
    y_pred, labels, input_length, label_length = args
    
    ctc_loss = K.ctc_batch_cost(
        labels, 
        y_pred, 
        input_length, 
        label_length
    )
    p = tensorflow.exp(-ctc_loss)
    gamma = 0.5
    alpha=0.25 
    return alpha*(K.pow((1-p),gamma))*ctc_loss

def load_easter_model(checkpoint_path):
    if checkpoint_path == "Empty":
        checkpoint_path = config.BEST_MODEL_PATH
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
    except Exception as e:
        print ("Unable to Load Checkpoint.: ", e)
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
    
import nltk
nltk.download('punkt')

import csv

def test_on_iam(lang,show=False, partition='validation', uncased=False, checkpoint="Empty"):
    
    print ("loading metdata...")
    training_data = data_loader(config.DATA_PATH, config.BATCH_SIZE, lang)
    validation_data = data_loader(config.DATA_PATH, config.BATCH_SIZE, lang)
    test_data = data_loader(config.DATA_PATH, config.BATCH_SIZE, lang)

    training_data.trainSet()
    validation_data.validationSet()
    test_data.testSet()
    charlist = training_data.charList
    print ("loading checkpoint...")
    print ("calculating results...")
    
    model = load_easter_model(checkpoint)
    char_error = 0
    total_chars = 0
    word_error = 0
    total_words = 0
    batches = 1
    
    while batches > 0:
        batches = batches - 1
        if partition == 'validation':
            print ("Using Validation Partition ....")
            imgs, truths, _ = validation_data.getValidationImage()
            print("Done..")
        else:
            print ("Using Test Partition")
            imgs,truths,_ = test_data.getTestImage()
        
        # initialize the list to store the predictions and the ground truth
        dict_list = []
        print ("Number of Samples : ",len(imgs))
        for i in tqdm(range(0,len(imgs))):
            img = imgs[i]
            truth = truths[i].strip(" ").replace("  "," ")
            output = model.predict(img)
            prediction = decoder(output, charlist)
            output = (prediction[0].strip(" ").replace("  ", " "))
            
            # append the prediction and the ground truth to the respective lists
            dict_list.append({"actual":truth,"pred":output})
    
            if uncased:
                char_error += edit_distance(output.lower(),truth.lower())
            else:
                char_error += edit_distance(output,truth)
                
            total_chars += len(truth)
            
            # tokenize the prediction and the ground truth and calculate the word error rate
            if uncased:
                pred_tokens = [token.lower() for token in output.split()]
                truth_tokens = [token.lower() for token in truth.split()]
            else:
                pred_tokens = output.split()
                truth_tokens = truth.split()
                
            word_error += edit_distance(pred_tokens, truth_tokens)
            total_words += len(truth_tokens)
            
            if show:
                print ("Ground Truth :", truth)
                print("Prediction [",edit_distance(output,truth),"]  : ",output)
                print ("*"*50)
                    
    results_df = pd.DataFrame(dict_list)
    results_df.to_csv(f"{lang}_val_small.csv")
            
    
    print ("Character error rate is : ",(char_error/total_chars)*100)
    print ("Word error rate is : ",(word_error/total_words)*100)
    

    
data_models = {"kannada": "/home/pageocr/easter2/Easter2/weights-kannada/Best_EASTER2--130--0.68.hdf5",
               "gujarati": "/home/pageocr/easter2/Easter2/weights-gujarati/BEST1_EASTER2--40--1.17.hdf5",
               "malayalam": "/home/pageocr/easter2/Easter2/weights-malaylam/BEST_EASTER2--28--0.95.hdf5",
               "tamil": "/home/pageocr/easter2/Easter2/weights-tamil/EASTER2--100--0.81.hdf5",
               "telugu": "/home/pageocr/easter2/Easter2/weights-telegu/best_telegu.hdf5",
               "devanagari": "/home/pageocr/easter2/Easter2/weights-hindi/weights/EASTER2--148--1.90.hdf5"}

for lang in data_models:
    test_on_iam(lang,checkpoint = data_models[lang])

# results_df = pd.DataFrame(results)
# results_df.to_json("./scores.json",orient='records')
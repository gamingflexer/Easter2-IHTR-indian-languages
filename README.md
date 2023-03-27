# Easter2.0: IMPROVING CONVOLUTIONAL MODELS FOR HANDWRITTEN TEXT RECOGNITION | IHTR DATASET | INDIAN LANGUAGES


[[[Arxiv-PDF-Link](https://arxiv.org/pdf/2205.14879.pdf)]]

### Overview
In this paper, we proposed a convolutional architecture for the task of handwritten text recognition that utilizes only 1D
convolutions, dense residual connections and a SE module. We also proposed a simple and effective data augmentation
technique-[’Tiling and Corruption (TACo)’](https://github.com/kartikgill/taco-box) useful for OCR/HTR tasks. We have presented experimental study on components of Easter2.0
architecture including dense residual connections, normalization choices, SE module, TACo variations and few-shot
training. Our work achieves SOTA results on IAM-Test set when training data is limited, also Easter2.0 has very
small number of trainable parameters compared to other solutions. The proposed architecture can be used in search of
smaller, faster and efficient OCR/HTR solutions when available annotated data is limited.

### How to use?

The following steps can help setting up Easter2 fast:

 - Download checkpoint from release, and put it inside ```/weigths``` directory if you want to finetune the older checkpoint or make it `False` in the config.py
 - Download you dataset in the data folder
 - Use the `custom_dataset.ipynb` to convert your dataset into the required format.
 - install requirements as per the file ```requirements.txt```
 - Modify ```/src/config.py``` as per your needs
 - run the ```train()``` function from ```/src/easter_model.py``` or just simply execute the `easter_model.py`
 - sample training and testing notebooks are given in ```/notebooks``` directory

#### Check GPU

```
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

## Owner & Citation

I have used the work of chaudhary2022easter2. If you find our work helpful, please cite the following:
```
@article{chaudhary2022easter2,
  title={Easter2. 0: Improving convolutional models for handwritten text recognition},
  author={Chaudhary, Kartik and Bali, Raghav},
  journal={arXiv preprint arXiv:2205.14879},
  year={2022}
}
```

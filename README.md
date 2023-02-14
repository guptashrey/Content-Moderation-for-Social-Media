# Content Moderation for Social Media
**Project by Andrew Bonafede, Shrey Gupta, and Shuai Zheng for Duke AIPI 540 Computer Vision Module**

## Project Description
Image content moderation is crucial in ensuring the safety and well-being of individuals who use digital platforms. With the widespread use of the internet, it's become easier to access and share images, both good and bad. This ease of access to visual content has also led to an increase in the spread of inappropriate, violent, and pornographic images. These types of images can be harmful and traumatizing for those who come across them, particularly children and young people.

By filtering out inappropriate, violent, and pornographic images, digital platforms can create a safer and more inclusive environment for all users. Moderating these types of images can also help prevent the spread of harmful beliefs and legal consequences for platform operators.

Our project focuses on providing a content moderation solution for social media. The aim is to classify social media images into three categories: violence, nsfw, and normal.

## About The Data
We identified three key image datasets that we believe will be beneficial in modeling our content moderation problem. These datasets are:

### NSFW Images Data
The NSFW images dataset used was collected from the [
EBazarov/nsfw_data_source_url](https://github.com/EBazarov/nsfw_data_source_urls) github repository. The repository contains links to 1,589,331 NSFW images across 159 different categories.

We wrote a python script to sample 10,000 image urls from this dataset and download the images. The script can be found in the `scripts` folder. The images were downloaded into the `data/images/nsfw` folder. As the images were large in size, we compressed them into a zip file and uploaded it to Box drive which can be accessed [here](https://duke.box.com/shared/static/81jx04bwnct82prwqh5ikm6xq9h8ey3t.zip).

### Neutral Images Data
The NSFW images dataset used was collected from the [
alex000kim/nsfw_data_scraper](https://github.com/alex000kim/nsfw_data_scraper) github repository. The repository contains links to 36,837 neutral images across different categories.

We wrote a python script to download all these images. The script can be found in the `scripts` folder. The images were downloaded into the `data/images/neutral` folder. As the images were large in size, we compressed them into a zip file and uploaded it to Box drive which can be accessed [here](https://duke.box.com/shared/static/xg0790jdcsk6mfbgpw59tqk0ym6sq9mf.zip).

### Violence Images Data
The violence images dataset used was collected manually using the [Google Images](https://images.google.com/) search engine. We searched for specific keywords such as agression, assault, hostility, riot, mob lynching etc. and downloaded the results using the Chrome Browser extension Fatkun.

As the images were large in size, we compressed them into a zip file and uploaded it to Box drive which can be accessed [here](https://duke.box.com/shared/static/g87440yf6v4pun482mcvon3kiloe758w.zip).

### Accessing the Data
As mentioned, the dataset used for training the models is uploaded to a Box drive. Feel free to use the links above. In case, you are unable to access them, please email Shrey Gupta at s.gupta[AT]duke[DOT]edu with your name and reason for accessing the data and we will be happy to provide access.

## Data Processing
Once the images were downloaded using the URLs, we wrote a python script to do the following:
- Remove corrupt images
- Remove duplicate images
- Resave the images after optimizing the quality to 80%

The processed data on which the models were trained can either be downloaded directly using the links above or the scripts to download and optimize the dataset can be found in the `scripts` folder and can be run as follows:

**1. Create a new conda environment and activate it:** 
```
conda create --name cv python=3.8
conda activate cv
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**3. Run the data download script:** 
```
python scripts/dataset/make_dataset.py
```
**4. Run the data optimize script:** 
```
python scripts/dataset/optimize_dataset.py
```

## Project Structure
The project data and codes are arranged in the following manner:

```
├── README.md                           <- description of project and how to set up and run it
├── requirements.txt                    <- requirements file to document dependencies
├── scripts                             <- directory for data processing, modelling and utility scripts
    ├── dataset
        ├── make_dataset.py             <- script to get data
        ├── optimize_dataset.py         <- script to check if the images are corrupt and save them at 80% quality
    ├── evaluation                      <- script to evaluate different models
        ├── eval_config.json
        ├── eval.py
        ├── eval_helper_functions.py
    ├── resnet                          <- script to train and test the resnet model
        ├── resnet_config.json
        ├── resnet_helper_functions.py
        ├── resnet_transfer_learning.py
    ├── vgg                             <- script to train and test the vgg model
        ├── vgg_config.json
        ├── vgg_helper_functions.py
        ├── vgg_transfer_learning.py
    ├── svm                             <- script to train and test the svm model
        ├── svm_config.json
        ├── svm_helper_functions.py
        ├── svm_training.py.py
├── models                              <- directory for trained models
├── data                                <- directory for project data
    ├── raw                             <- directory for raw data or script to download
    ├── processed                       <- directory to store processed data
    ├── outputs                         <- directory to store any output data
├── scratch                             <- directory to store any intermediate and scratch files used
├── notebooks                           <- directory to store any exploration notebooks used
├── .gitignore                          <- git ignore file
```

&nbsp;
## Model Training and Evaluation
A non deep learning and a deep learning modelling approach were used to accurately identify and categorize social media images based on their content. The non deep learning approach used Random Forest and SVM models whereas the deep learning approach used VGG and Resnet model architectures.
### Non-Deep Learning Models

### SVM
To avoid the bottle neck of being able to train the model on a small dataset, a SVM (Support Vector Machine) model was trained. The PyTorch implementation of SVM was used to be able to use the whole dataset to train the model.

But due to the high dimensionality of the image data, the SVM model was not able to perform well. The model was able to achieve an accuracy of 40% on the test set. This is because SVMs are linear classifiers and are not able to capture non-linear relationships in the data. This is where deep learning models come in handy.

Following are the steps to run the code and train a SVM model:

**1. Create a new conda environment and activate it:** 
```
conda create --name cv python=3.8
conda activate cv
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```

**3. Tweak the model parameters [OPTIONAL]:** 
```
nano scripts/svm/svm_config.json
```

**4. Run the training script:** 
```
python scripts/svm/svm_training.py
```
&nbsp;
### Deep Learning Models
### VGG
A VGG19 model was trained on the dataset. Transfer learning was performed on the model in 2 ways:
- Case 1: VGG19_bn with all layers trainable
- Case 2: VGG19_bn with just the last fully connected layer trainable

The model was trained for 10 epochs in both the cases and the training and validation loss and accuracy were plotted. The model in case 1 was able to achieve an accuracy of 74% on the test set while the model in case 2 achieved 89%.

Following are the steps to run the code and train a VGG model:

**1. Create a new conda environment and activate it:** 
```
conda create --name cv python=3.8
conda activate cv
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```

**3. Tweak the model parameters [OPTIONAL]:** 
```
nano scripts/vgg/vgg_config.json
```

**4. Run the training script:** 
```
python scripts/vgg/vgg_transfer_learning.py
```
### Resnet

The SOTA model in image classification - Resnet was trained on the dataset in the following variations:
- Case 1: Resnet18 with all layers trainable
- Case 2: Resnet18 with just the last fully connected layer trainable
- Case 3: Resnet50 with all layers trainable
- Case 4: Resnet50 with just the last fully connected layer trainable
- Case 5: Resnet152 with all layers trainable
- Case 6: Resnet152 with just the last fully connected layer trainable

We trained three models for comparison. We started with a simple image classifier - Resnet18. The labelled images were split into train and test datasets and loaded in Pytorch Dataset Objects. A pretrained Resnet18 model was loaded and its fully connected head was stripped off to perform transfer learning. The new trasnfer learned resnet takes in input images of size 640x640 and gives the predicted probability of presence of each class in the image.

The results of the model were as follows:
| Model     | Mode             | Accuracy (Test) |
| --------- | ---------------- | :-------------: |
| Resnet18  | Train all layers | 85%             |
| Resnet18  | Train FC layer   | 84%             |
| Resnet50  | Train all layers | 87%             |
| Resnet50  | Train FC layer   | 88%             |
| Resnet152 | Train all layers | 89%             |
| Resnet152 | Train FC layer   | 91%             |

Following are the steps to run the code and train a Resnet model:

**1. Create a new conda environment and activate it:** 
```
conda create --name cv python=3.8
conda activate cv
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```

**3. Tweak the model parameters [OPTIONAL]:** 
```
nano scripts/resnet/resnet_config.json
```

**4. Run the training script:** 
```
python scripts/resnet/resnet_transfer_learning.py
```


## Content Moderation Application (Streamlit):
* Refer to the [README.md](https://github.com/guptashrey/Content-Moderation-for-Social-Media/blob/st/README.md) at this link to run the streamlit based web application or access it [here](https://guptashrey-content-moderation-for-1--content-moderation-5y11hh.streamlit.app/).
* You can find the code for the stremalit web-app [here](https://github.com/guptashrey/Content-Moderation-for-Social-Media/tree/st)

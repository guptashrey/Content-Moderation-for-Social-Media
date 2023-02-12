# Content Moderation for Social Media
Our project focuses on providing a content moderation solution for social media. The aim is to classify social media images into three categories: violence, porn, and normal.

## Model Training
We have trained a machine learning model to accurately identify and categorize social media images based on their content. The model is trained on a large dataset of social media images and uses advanced techniques to accurately classify each image.

## Install dependencies
> pip install .

## Run the project (get data, build features, train model) 
> python main.py
## Deep Learning Model (Transfer learning with ResNet18)
> Located in ./scrips/model.py
```
def train_predict_dl_model(image_path,model_save_path):

    #   ...code here...

```


## Non Deep Learning Model (Random Forest, Decision Tree)
> Located in ./scrips/model.py
```
def train_predict_non_dl_model(image_path,model_save_path):

   #   ...code here...
```


## Project repo structure

```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── setup.py                <- script to set up project (get data, build features, train model)
├── main.py [or main.ipynb] <- main script/notebook to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to get data
    ├── build_features.py   <- script to run pipeline to generate features 
    ├── model.py            <- script to train model and predict
├── models                  <- directory for trained models
├── data                    <- directory for project data
    ├── raw                 <- directory for raw data or script to download
    ├── processed           <- directory to store processed data
    ├── outputs             <- directory to store any output data
├── notebooks               <- directory to store any exploration notebooks used
├── .gitignore              <- git ignore file
```


## Features
* Accurate classification of news images into violence, porn, and normal categories.
* Used transfer learning to train a deep learning model.
* Compared deep learning and non-deep learning models (Random Forest, Decision Tree) to determine which model is more effective.

## Conclusion
After comparison, the deep learning has better performance in image classification. We believe that our content moderation solution will help create a safer and more enjoyable online environment for social media users. By removing violent and pornographic content, we aim to create a space where people can comfortably share and consume news without fear of being exposed to inappropriate material.



# Content Moderation for Social Media
Our project focuses on providing a content moderation solution for social media. The aim is to classify social media images into three categories: violence, porn, and normal.

## Model Training
We have trained a machine learning model to accurately identify and categorize social media images based on their content. The model is trained on a large dataset of social media images and uses advanced techniques to accurately classify each image.

## Install dependencies
> pip install .

## Project repo structure

```
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile [OPTIONAL]     <- setup and run project from command line
├── setup.py                <- script to set up project (get data, build features, train model)
├── main.py [or main.ipynb] <- main script/notebook to run project / user interface
├── scripts                 <- directory for pipeline scripts or utility scripts
    ├── make_dataset.py     <- script to get data [OPTIONAL]
    ├── build_features.py   <- script to run pipeline to generate features [OPTIONAL]
    ├── model.py            <- script to train model and predict [OPTIONAL]
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
* Our content filter solution can be easily integrated with existing social media platforms to provide a safer and more enjoyable user experience. 
* The filter works in real-time, analyzing each article images as it is posted and categorizing it accordingly.


## Conclusion
We believe that our content moderation solution will help create a safer and more enjoyable online environment for social media users. By removing violent and pornographic content, we aim to create a space where people can comfortably share and consume news without fear of being exposed to inappropriate material.



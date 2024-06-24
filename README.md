# technical-test

### Description
This project consists of two parts. 

The first part involves training an image classifier for cats and dogs after performing pre-processing and testing different classifiers and hyperparameters. The goal is to select the best classifier based on the accuracy evaluation metric.

The second part involves the analysis and forecasting of a time series of bicycle demand. The dataset is available at the following link:

#### **Dataset[Dataset-part2](https://www.kaggle.com/competitions/bike-sharing-demand/data)**

### Dev Instructions:
To clone the repository navigate to the destination folder in the terminal and clone the repository type the following on terminal:

For macOS/Windows:

```
git clone <repository_url>
```

Once the repository is cloned install Miniconda on your local machine.

#### **Install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)**

After install Miniconda locally, navigate through the terminal to the cloned repository, execute the yaml file (env.yml) with the following command:

For macOS/Windows:

```
conda env create -f env.yml
````

### Part 1. Image Classification (Cat vs Dog):

In the `technical-test/part1` directory, you will find:
- The `data` folder, which contains the images used for training, validation, and testing.
- The `models` folder, which contains the trained models with the best hyperparameter configuration selected during validation.
- The `production` folder, which contains the files needed for production deployment, including the Docker container setup to generate the Docker image for the project, the `requirements.txt` file listing the packages used in the project, and the `predict.py` script for predicting new images using the best selected model (InceptionV3).
- The `part1-classification.ipynb` notebook, which provides a detailed walkthrough of the entire process of developing the image classification model. This includes data preprocessing steps, exploratory data analysis (EDA), testing various classifiers, tuning hyperparameters, and evaluating the performance of different models. The notebook documents the rationale behind each decision, the methodologies used for improving model accuracy, and the final selection of the best model for deployment.

#### Build the Docker Image

Run the following command in the `production` directory to build the Docker image:
```bash
docker build -t image-classification-app .
```

### Part 2. Analysis and Forecasting (Bicycle demand)

In the `technical-test/part2` directory, you will find:
- The `data` folder, which contains the dataset used for analysis and forecasting
- The `models` folder, which contains the trained models with the best hyperparameter configuration selected during training.
- The 'part2-multivariate-timeseries.ipynb' notebook which contains the analysis of the dataset and the training process of the different models and the corresponding prediction of the one with the best results.
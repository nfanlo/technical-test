# technical-test

## Description
This project consists of two parts. 

The first part 

The second part involves the analysis and forecasting of a time series of bicycle demand based on MAE and MSE. The dataset is available at the following link:

#### **Download [Dataset-part2](https://www.kaggle.com/competitions/bike-sharing-demand/data)**

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
```

### Part 1. 

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
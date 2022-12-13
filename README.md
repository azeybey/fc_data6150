
Flower Classification using TPU - Data 6150 Class Term Project

I build a machine learning model that identifies the type of flowers in a dataset of images.

# Deployed Data Science Project Sample


## Introduction
The objective of this project is to build a machine learning model that identifies the type of flowers in a dataset of images. 

This project is inspired by the [the Kaggle TPU Competition](https://www.kaggle.com/competitions/tpu-getting-started/) [3]. 

I used Tensor Processing Units to accelerate my project. TPUs are powerful hardware accelerators specialized in deep learning tasks. They were developed (and first used) by Google to process large image databases, such as extracting all the text from Street View. 

Furtunetely, Kaggle offers accessing to 30 hours of free TPU time every week for its users. So, I deploy my notebook to Kaggle to utilize its processing power.

## Selection of Data

The model processing and training are conducted using a Jupyter Notebook and is available [here](https://github.com/memoatwit/dsexample/blob/master/Insurance%20-%20Model%20Training%20Notebook.ipynb).

When used with TPUs, datasets need to be stored in a Google Cloud Storage bucket. You can use data from any public GCS bucket by giving its path. The following retrieves the GCS path for my dataset.

<img width="526" alt="Screen Shot 2022-12-13 at 13 07 06" src="https://user-images.githubusercontent.com/12528641/207411220-ae27e383-e1a5-4345-88fa-8eacfa165ec9.png">

Google Cloud Store Path :  gs://kds-815d280b11fe375202da79512bafaf620b4095ed95aae36da10b22b1

The data has over 1,300 samples with 6 features: age, sex, BMI, no of children, smoker, region.  
The objective is to predict hospital charges.
The dataset can found online at [git](https://github.com/stedy/Machine-Learning-with-R-datasets)[4]. Many example solutions and analysis can be found at [kaggle](https://www.kaggle.com/mirichoi0218/insurance) [5]

Data preview: 
![data screenshot](./insurance_data.png)


Note that data has categorical features in 3 cols: sex, smoker, and region.
I used OneHotEncoder/ColumnTransformer on these features and kept the rest of the features as is.
When using a linear regression model, the learning accuracy shoots up from 12% to 75% with this transformation. 

We then create a pipeline for automating the process for new data. I also experimented with different imputer strategies for missing data and added second-degree polynomial features for both the numeric and categorical data. The shown values are obtained by performing a grid search over these arguments. 
Pipeline preview: 
![pipeline screenshot](./pipeline.png)

I finally saved the model via joblib to be used for predictions by the web app. 

## Methods

Tools:
- NumPy, SciPy, Pandas, and Scikit-learn for data analysis and inference
- Streamlit (st) for web app design
- GitHub and Heroku for web app deployment and hosting/version control
- VS Code as IDE

Inference methods used with Scikit:
- linear regression model
- Features: OneHotEncoder/ColumnTransformer, StandardScaler, PolynomialFeatures and SimpleImputer for missing values
- Pipeline to tie it all together
- GridSearchCV for hyperparameter tuning

## Results
The app is live at https://ds-example.herokuapp.com/
It allows for online and batch processing as designed by the pycaret post:
- Online: User inputs each feature manually for predicting a single insurance cost
![online screenshot](./online.png)
- Batch: It allows the user to upload a CSV file with the 6 features for predicting many instances at once. 
  - An [X_test.csv](./X_test.csv) is provided as a batch processing sample. Corresponding insurance prices are available at [y_test.csv](./y_test.csv)
![batch screenshot](./batch.png)

I am not adding any visualizations to this example, though st supports it. Couple good examples are [here](https://share.streamlit.io/tylerjrichards/book_reco/books.py) and [here](https://share.streamlit.io/streamlit/demo-uber-nyc-pickups/)

## Discussion
Experimenting with various feature engineering techniques and regression algorithms, I found that linear regression with one-hot encoding provided one of the highest accuracies despite its simpler nature. Across all these trials, my training accuracy was around 75% to 77%. Thus, I decided the deploy the pipelined linear regression model. The data was split 80/20 for testing and has a test accuracy of 73%. 

I looked at some kaggle notebooks studying this problem and found this to be an acceptable level of success for this dataset. I am interested in analyzing the training data further to understand why a higher accuracy can't be easily achieved, especially with non-linear kernels. 

One unexpected challenge was the free storage capacity offered by Heroku. I experimented with various versions of the libraries listed in `requirements.txt` to achieve a reasonable memory footprint. While I couldn't include the latest pycaret library due to its size, the current setup does include TensorFlow 2.3.1 (even though not utilized by this sample project) to illustrate how much can be done in Heroku's free tier: 
```
Warning: Your slug size (326 MB) exceeds our soft limit (300 MB) which may affect boot time.
```

## Summary
This sample project deploys a supervised regression model to predict insurance costs based on 6 features. After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 73%. 

The web app is designed using Streamlit, and can do online and batch processing, and is deployed using Heroku and Streamlit. The Heroku app is live at https://ds-example.herokuapp.com/.
 
Streamlit is starting to offer free hosting as well. The same repo is also deployed at [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/memoatwit/dsexample/app.py)  
More info about st hosting is [here](https://docs.streamlit.io/en/stable/deploy_streamlit_app.html).


## References
[1] [GitHub Integration (Heroku GitHub Deploys)](https://devcenter.heroku.com/articles/github-integration)

[2] [Streamlit](https://www.streamlit.io/)

[3] [The pycaret post](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)

[4] [Insurance dataset: git](https://github.com/stedy/Machine-Learning-with-R-datasets)

[5] [Insurance dataset: kaggle](https://www.kaggle.com/mirichoi0218/insurance)

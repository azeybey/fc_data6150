
## Flower Classification using TPU - Data 6150 Class Term Project

I build a machine learning model that identifies the type of flowers in a dataset of images.

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

Number of training images : 12753 
Number of validation images : 3712 
Number of test images : 7382

The data has 12753 training images, 3712 validation images and 7382 test images. 
Training dataset has (12753, 512, 512, 3) images data and (12753,) label data.
Validation dataset has (3712, 512, 512, 3) images data and (3712,) label data. 
Test dataset has (7382, 512, 512, 3) images data and (7382,) id data. 

The objective is to predict flower type(label data) of test images.

Data preview: 

<img width="838" alt="Screen Shot 2022-12-13 at 13 24 37" src="https://user-images.githubusercontent.com/12528641/207414730-cfbb7dde-8241-462a-9b9a-5f5812dae503.png">

## Methods

Tools:
- NumPy, Pandas, Pyplot and Tensorflow for data analysis and inference
- GitHub for hosting/version control

Inference methods used with Tensorflow:
- Transfer Learning
- Convolutional Neural Network - Deep Learning
- Pre-trained Model -> Fine Tune -> Re-train
- Adaptive Learning Rate

<img width="232" alt="Screen Shot 2022-12-13 at 00 27 28" src="https://user-images.githubusercontent.com/12528641/207415810-f52189ce-cb9a-4137-b236-cd536c510ac2.png">

## Results

The code is at https://www.kaggle.com/code/abdurrahmanzeybey/flower-classification-data-6150

Training Process:
During 15 iteration of training, learning rate starts at 0.000050 and ends at 0.000019. Training and validation accuracy improvement:

<img width="378" alt="Screen Shot 2022-12-13 at 13 38 26" src="https://user-images.githubusercontent.com/12528641/207417492-fc1073cd-9521-42c0-8e54-c8e8860069b0.png">

Prediction and Submission:
Test data comes with unlabeled, so I predict its labels and submit them to the Kaggle.

<img width="185" alt="Screen Shot 2022-12-13 at 13 40 18" src="https://user-images.githubusercontent.com/12528641/207417839-9fc6586b-c17e-4afe-a92f-6f8962c1790a.png">

Kaggle evaluates my submission on macro F1 score. My score is calculated as follows:
<img width="263" alt="Screen Shot 2022-12-13 at 13 41 46" src="https://user-images.githubusercontent.com/12528641/207418107-7e875c96-1469-4f95-ae84-3aaba0792b61.png">

After 4 submission, I achieve 0.92102 score.
<img width="1217" alt="Screen Shot 2022-12-13 at 13 44 09" src="https://user-images.githubusercontent.com/12528641/207418520-dc1527a2-9f2a-4d93-86c2-d4c65de7db0b.png">

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

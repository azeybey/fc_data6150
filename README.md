
## Flower Classification using TPU - Data 6150 Class Term Project

This project is developed for final project of Data Science Foundation class at Wentworth Institution of Technologies.

## Introduction
The objective of this project is to build a machine learning model that identifies the type of flowers in a dataset of images. 

This project is inspired by the [the Kaggle TPU Competition](https://www.kaggle.com/competitions/tpu-getting-started/) [1]. 

I used Tensor Processing Units to accelerate my project. Tensor Processing Unit (TPU) is an AI accelerator application-specific integrated circuit (ASIC) developed by Google for neural network machine learning, using Google's own TensorFlow software [2]. They were developed (and first used) by Google to process large image databases, such as extracting all the text from Street View. 

Furtunetely, Kaggle offers accessing to 30 hours of free TPU time every week for its users. So, I deploy my notebook to Kaggle to utilize its processing power.

## Selection of Data

The model processing and training are conducted using a Jupyter Notebook and is available [here](https://github.com/azeybey/fc_data6150/blob/main/flower-classification-data-6150.ipynb).

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

Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.[3] For example, knowledge gained while learning to recognize cars could apply when trying to recognize flowers. 

<img width="232" alt="Screen Shot 2022-12-13 at 00 27 28" src="https://user-images.githubusercontent.com/12528641/207415810-f52189ce-cb9a-4137-b236-cd536c510ac2.png">


<img width="517" alt="Screen Shot 2022-12-13 at 13 59 15" src="https://user-images.githubusercontent.com/12528641/207421408-d154d16d-c32e-49e7-8dea-f7125a0cff3a.png">


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
Experimenting with various CNN models, I found that TensorFlow Xception model provided one of the highest accuracies despite its simpler nature. It is pre-trained and is provided to us through the Keras Applications Library. It was originaly trained 350 Millions images that included 17000 classes. My goal was to change this model to classify photos of flowers into their species.

I looked at some kaggle notebooks studying this problem and found this to be an acceptable level of success for this dataset. I am interested in analyzing the training data further to understand why a higher accuracy can't be easily achieved.

One unexpected challenge was the size of the dataset. Kaggle provides us 12753 training and 3712 validation images which are small compare to the big model, like Xception was trained 350 Millions of images as I mentioned before. 

On the other hand, if I had enough time, I could work on data augmentation matrices that could be applied to the training images to enrich the data. It would increase the level of success.

## Summary

This project deploys a pre-trained CNN - Deep Learnin model called Xception to predict flowers type based on given images. After experimenting with various feature engineering techniques, the deployed model's testing accuracy hovers around 92%. 

The model is fine-tuned and re-trained with given training and validation images. The code is deployed to Kaggle Competition page at https://www.kaggle.com/competitions/tpu-getting-started/overview.


## References
[1] [https://www.kaggle.com/competitions/tpu-getting-started/]

[2] [https://en.wikipedia.org/wiki/Tensor_Processing_Unit]

[3] [https://en.wikipedia.org/wiki/Transfer_learning]


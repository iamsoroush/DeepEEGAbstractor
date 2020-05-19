# Deep EEG Abstractor (D-EEG-A)

In this project, we aim to predict whether a patient diagnosed with MDD (Major Depressive Disorder) will response to SSRI-based medications.  I have proposed a method called DeepEEGAbstractor, which is a temporal-CNN-based neural network for classification of multi-variate resting state EEG signals. First the input signal processed to obtain a set more abstract features by using 4 consecutive specially designed temporal convolution blocks. Then with the help of a temporal attention mechanism, model can classify input signals of any length, without the need of making the incoming samples the same length as the training samples. Also the input channel-wise dropout, makes the model robust against the loss of some input channels. And at the end, all the results can be re-produced using google colab.

Here's the model's architecture:
![Alt text](https://github.com/iamsoroush/DeepEEG/blob/master/deep_eeg_arch.jpg)


To reproduce the results, do the following:
1. Create a copy from [*prepare_data*](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/prepare_data.ipynb) in your google drive.
2. Run all cells within *Copy of prepare_data.ipynb* one-by-one.
2. Create a copy from a *cv_....ipynb* file in your google drive.
4. Run cells one-by-one, and you got the results!


# Results
## Overall Performance
#### Responder vs. Non Responder: Balanced dataset (5time-kfold CV)
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/balanced-rnr.bmp)

#### Responder vs. Non Responder: Cross-subject generalisation (10time-10fold CV)
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/cs-rnr.bmp)

#### Healthy vs. MDD: Balanced dataset (5time-5fold CV)
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/balanced-hmdd.bmp)

#### Healthy vs. MDD: Cross-subject generalisation (5time-5fold CV)
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/cs-hmdd.bmp)


## Embeddings Visualization: Fixed len test set [4s]
#### After 2 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-2epochs-balanced-fixed4s.gif)

#### After 5 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-5epochs-balanced-fixed4s.gif)

#### After 10 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-10epochs-balanced-fixed4s.gif)

#### After 50 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-50epochs-balanced-fixed4s.gif)

## Embeddings Visualization: Variable length test set [4s, 10s]
#### After 1 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/varlen-epoch1.gif)

#### After 5 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/varlen-epoch5.gif)

#### After 10 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/varlen-epoch10.gif)

#### After 40 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/varlen-epoch40.gif)

#### Distribution of data instance length after 40 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/durations-epoch50.gif)

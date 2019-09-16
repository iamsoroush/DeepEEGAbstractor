# DeepEEGAbstractor
Deep EEG Abstractor


Do the following:
1. Download dataset from [here](https://figshare.com/articles/EEG_Data_New/4244171) to your google drive
2. Make numpy arrays from .edf files and place them under _data_path_ using following code:

3. Open a notebook from _notebooks_ folder in google colab.
4. Run the notebook cell-by-cell
5. And now you have the results!


# Results
## Overall Performance
### Responder vs. Non Responder: Balanced dataset (5time-kfold CV)

...

### Responder vs. Non Responder: Cross-subject generalisation (10time-10fold CV)

...

### Healthy vs. MDD: Balanced dataset (5time-5fold CV)

...

### Healthy vs. MDD: Cross-subject generalisation (5time-5fold CV)

...


## Embeddings Visualization: Fixed len test set [4s]
### After 2 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-2epochs-balanced-fixed4s.gif)

### After 5 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-5epochs-balanced-fixed4s.gif)

### After 10 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-10epochs-balanced-fixed4s.gif)

### After 50 epochs
![Alt text](https://github.com/iamsoroush/DeepEEGAbstractor/blob/master/results/rnr-50epochs-balanced-fixed4s.gif)

## Embeddings Visualization: Variable length test set [4s, 10s]
### After 2 epochs
![Alt text]()

### After 5 epochs
![Alt text]()

### After 10 epochs
![Alt text]()

### After 50 epochs
![Alt text]()

# Remaining-Useful-Life-Prediction
Remaining Useful Life prediction of machinery using a novel data wrangling method and CNN-LSTM network for prediction

Creatensp.py : A novel data wrangling method that changes raw (vibration) data into an image that can used for image analysis and different computer vision research.

Nspcnnlstm_hi.py : Using NSP images for feature extraction. The data first goes through CNN layers in which important features are extracted. Then it goes through LSTM layers which remembers sequences of past data to make accurate RUL predictions.

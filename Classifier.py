from tensorflow import keras
import tensorflow as tf
import tensorflow_text as text
import pandas as pd
import re
import numpy as np

from keras import backend as K


def balanced_recall(y_true, y_pred):
    """This function calculates the balanced recall metric
    recall = TP / (TP + FN)
    """
    recall_by_class = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        recall_by_class = recall_by_class + recall
    return recall_by_class / y_pred.shape[1]


def balanced_precision(y_true, y_pred):
    """This function calculates the balanced precision metric
    precision = TP / (TP + FP)
    """
    precision_by_class = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        precision_by_class = precision_by_class + precision
    # return average balanced metric for each class
    return precision_by_class / y_pred.shape[1]


def balanced_f1_score(y_true, y_pred):
    """This function calculates the F1 score metric"""
    precision = balanced_precision(y_true, y_pred)
    recall = balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# load model
model = keras.models.load_model("Prototype3/models/text_classifier_v1",
                                custom_objects={'balanced_recall': balanced_recall,
                                                'balanced_precision': balanced_precision,
                                                'balanced_f1_score': balanced_f1_score})


def predict_class(message):
    '''predict class of input text
    Args:
    - reviews (list of strings)
    Output:
     - class (list of int)
    '''
    print("the prediction is: " + str(model.predict(message)))
    print(model.predict(message)[0][1])
    #if probability > 0.8 return [np.argmax(pred) for pred in model.predict(message)]
    if model.predict(message)[0][1] > 0.8:
        return [np.argmax(pred) for pred in model.predict(message)]
    elif model.predict(message)[0][0] > 0.5:
        return [np.argmax(pred) for pred in model.predict(message)]



print(predict_class(["hello"]))
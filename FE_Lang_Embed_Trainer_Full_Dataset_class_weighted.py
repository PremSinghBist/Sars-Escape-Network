import sys
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5"

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    concatenate, Activation, Dense, Embedding, LSTM, Reshape)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.models as m
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
np.random.seed(1)
import math
import pandas as pd

import AdditionalSignficantEscapeSeqGenerator as ASE
import featurizer as FZ
import DataAnalyzer

import FE_Lang_Embed_Trainer_Full_Dataset as FD

MODEL_PATH = "model/integrated_model_class_weighted"
BASE_ANALYSIS_PATH = "data/analysis/integrated_class_weighted"

def get_dense_model():
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense 
    from tensorflow.keras.layers import BatchNormalization 
    INPUT_FEATURE_SIZE = 512
    model = Sequential()
    model.add(layers.Input(INPUT_FEATURE_SIZE, ))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    optmz = tf.keras.optimizers.Adam(
        name = "Adam",
        learning_rate = 0.00001
    )

    

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optmz,  metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()
    print(model.optimizer.get_config())
    
    return model
def get_class_weights(neg_count=0, pos_count=0):
    total = neg_count + pos_count
    weight_for_0 = (1 / neg_count) * (total / 2.0)
    weight_for_1 = (1 / pos_count) * (total / 2.0)
    return {0: weight_for_0, 1: weight_for_1 }

def get_datasets(non_sig_train, non_sig_test, sig_train_features, sig_test):
    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 40166 
    #NOn Sig train test features
    non_sig_train_len = non_sig_train.shape[0]
    non_sig_features_train_output = np.zeros( (non_sig_train_len, 1))
    print(f" non_sig_train shape: {non_sig_train.shape} Non Sig test shape {non_sig_test.shape} ")
    non_sig_train_dataset = tf.data.Dataset.from_tensor_slices((non_sig_train, non_sig_features_train_output))

    non_sig_test_len = non_sig_test.shape[0]
    non_sig_test_features_output = np.zeros( (non_sig_test_len, 1))
    non_sig_test_dataset = tf.data.Dataset.from_tensor_slices((non_sig_test, non_sig_test_features_output))
    

    #Over sample sig train features 
    sig_train_len =  sig_train_features.shape[0]
    print("Sig Train shape: ",sig_train_features.shape )
    sig_train_features_output = np.ones( (sig_train_len, 1) )
    sig_train_dataset = tf.data.Dataset.from_tensor_slices((sig_train_features, sig_train_features_output))

 
    sig_test_len =  sig_test.shape[0]
    sig_test_features_output = np.ones( (sig_test_len, 1) )
    sig_test_dataset = tf.data.Dataset.from_tensor_slices((sig_test, sig_test_features_output))

    combined_train_ds = sig_train_dataset.concatenate(non_sig_train_dataset)
    combined_train_ds = combined_train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset_cardinality = combined_train_ds.cardinality().numpy()
    print(f"Combined Train features Cardinality: {train_dataset_cardinality} ") #No of rows returns 

    combined_test_ds = sig_test_dataset.concatenate(non_sig_test_dataset)
    combined_test_ds = combined_test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset_cardinality = combined_test_ds.cardinality().numpy()
    print(f"Combined Train features Cardinality: {train_dataset_cardinality} ") #No of rows returns 

    return combined_train_ds, combined_test_ds

def execute_aggregate_network():
    non_sig_train  = FD.get_features("data/additional_escape_variants/gen/non_sig_combined_windows_train.csv")
    non_sig_test  = FD.get_features("data/additional_escape_variants/gen/non_sig_combined_windows_test.csv")

    sig_train_features  =  FD.get_features("data/additional_escape_variants/gen/sig_combined_windows_train.csv")
    sig_test =   FD.get_features("data/additional_escape_variants/gen/sig_combined_windows_test.csv")

   
    model  = get_dense_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  

    train_ds, val_ds = get_datasets(non_sig_train, non_sig_test, sig_train_features, sig_test)

    neg_count = non_sig_train.shape[0] + non_sig_test.shape[0]
    pos_count = sig_train_features.shape[0] + sig_test.shape[0]
    cls_wts = get_class_weights(neg_count, pos_count)

    model.fit(train_ds,  epochs=30, validation_data=val_ds, class_weight=cls_wts,  callbacks= [callback], verbose=1) #verbose 1 = Progress bar
    
    model.save(MODEL_PATH)

def plot_greany_test_predictions(targets, y_pred):
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_AUC_class_weighted.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_pr_curve_class_weighted.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_confusion_matrix__class_weighted.png')

def analyze_greany():
    
    model = m.load_model(MODEL_PATH)
    model.summary()
    sig_data = FD.get_features("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv")
    non_sig_data =  FD.get_features("data/additional_escape_variants/gen/greany_non_sig_combined_windowed_seqs.csv")

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    print("Evaluating  greany Combined ")
    model.evaluate(features, targets)
    print("Evaluating  greany Sig only ")
    model.evaluate(sig_data, sig_features_output)
    print("Evaluating  greany Non Sig only ")
    model.evaluate(non_sig_data, nonsig_features_output)

    #Analyze greany predictions | plot AUC_ROC / Confusion Matrix / Precision Recall
    y_preds = model.predict(features)
    plot_greany_test_predictions(targets, y_preds)

if __name__ == "__main__":
    execute_aggregate_network()
    analyze_greany()

    pass

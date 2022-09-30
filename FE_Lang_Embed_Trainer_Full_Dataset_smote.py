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

MODEL_PATH = "model/integrated_oversample_smote"
BASE_ANALYSIS_PATH = "data/analysis/integrated_oversample_smote"

def get_dense_model():
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense 
    from tensorflow.keras.layers import BatchNormalization 
    INPUT_FEATURE_SIZE = 512
    model = Sequential()
    model.add(layers.Input( shape = (INPUT_FEATURE_SIZE, ) ))
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

def generate_smote_samples(X, y):
    from imblearn.over_sampling import SMOTE 
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_res = np.reshape(X_res , (-1, 512))
    return X_res, y_res

def save_embedded_features():
    non_sig_train  = FD.get_features("data/additional_escape_variants/gen/non_sig_combined_windows_train.csv")
    sig_train_features  =  FD.get_features("data/additional_escape_variants/gen/sig_combined_windows_train.csv")

    sig_test =   FD.get_features("data/additional_escape_variants/gen/sig_combined_windows_test.csv")
    non_sig_test  = FD.get_features("data/additional_escape_variants/gen/non_sig_combined_windows_test.csv")

    greany_sig_data = FD.get_features("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv")
    greany_non_sig_data =  FD.get_features("data/additional_escape_variants/gen/greany_non_sig_combined_windowed_seqs.csv")

    np.savez('data/embed_features.npz', 
                NON_SIG_TRAIN= non_sig_train, SIG_TRAIN =sig_train_features, 
                NON_SIG_TEST= non_sig_test, SIG_TEST=sig_test,
                GREANY_SIG = greany_sig_data, NON_SIG_GREANY = greany_non_sig_data
                  )
    print("All Features Saved Successfully")

def load_embedded_features():
    embed_features = np.load('data/embed_features.npz')
    return embed_features




def execute_aggregate_network():
    BATCH_SIZE = 24
    embed_features = load_embedded_features()
    non_sig_train  = embed_features['NON_SIG_TRAIN']
    non_sig_train_len = non_sig_train.shape[0]
    non_sig_features_train_output = np.zeros( (non_sig_train_len, 1))
    print("NOng Sig Train shape: ",non_sig_train.shape )

    sig_train_features  =  embed_features['SIG_TRAIN']
    sig_train_len =  sig_train_features.shape[0]
    print("Sig Train shape: ",sig_train_features.shape )
    sig_train_features_output = np.ones( (sig_train_len, 1) )

    X = np.concatenate( (non_sig_train, sig_train_features ) , axis=0 )
    y = np.concatenate( (non_sig_features_train_output, sig_train_features_output), axis=0 )

    X_resampled, y_resampled  = generate_smote_samples(X, y)
    X_resampled = np.reshape(X_resampled, (-1, 512))
    y_resampled = y_resampled.reshape( (-1, 1))
    print("Training Data shape after smote resampling: ", X_resampled.shape )

    

    train_set = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))
    train_set.batch(BATCH_SIZE)


    sig_test =   embed_features['SIG_TEST']
    non_sig_test  = embed_features['NON_SIG_TEST']

    print(f"NonSig test shape: {non_sig_test.shape} Shape[0] {non_sig_test.shape[0]}")
    

    ts = np.concatenate( (sig_test, non_sig_test), axis=0)
    ts = np.reshape(ts, (-1, 512))
    ts_output_0 = np.ones( (sig_test.shape[0], 1) )
    ts_output_1 = np.zeros( (non_sig_test.shape[0], 1) )
    ts_output = np.concatenate( (ts_output_0, ts_output_1), axis=0)
    ts_output = np.reshape(ts_output, (-1, 1))
    test_set = tf.data.Dataset.from_tensor_slices((ts, ts_output))
    test_set.batch(BATCH_SIZE)


    print("*******************")
    print(f"Shape of train set: {X_resampled.shape}, output: {y_resampled.shape}")
    print(f"Shape of test set: {ts.shape}, output: {ts_output.shape}")
    
    print("*******************************")

   
    model  = get_dense_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  
    #model.fit(train_set,  epochs=30, validation_data=test_set, callbacks= [callback], verbose=1) #verbose 1 = Progress bar
    model.fit(X_resampled,y_resampled ,  epochs=30, validation_data=(ts, ts_output), callbacks= [callback], verbose=1) #verbose 1 = Progress bar
    model.save(MODEL_PATH)

    #Analyze greany 
    greany_sig = embed_features['GREANY_SIG']
    greany_non_sig = embed_features['NON_SIG_GREANY']
    analyze_greany(greany_sig, greany_non_sig)

def plot_greany_test_predictions(targets, y_pred):
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_AUC_class_weighted.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_pr_curve_class_weighted.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_confusion_matrix__class_weighted.png')

def analyze_greany(sig_data, non_sig_data):
    
    model = m.load_model(MODEL_PATH)
    model.summary()

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
    #model  = get_dense_model()
    #save_embedded_features()
    execute_aggregate_network()

    pass

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
from transformers import FEATURE_EXTRACTOR_MAPPING
np.random.seed(1)
import math
import pandas as pd

import AdditionalSignficantEscapeSeqGenerator as ASE
import featurizer as FZ
import DataAnalyzer
MODEL_PATH = "embedding_full_model/checkpoints/bilstm/bilstm_256-04.hdf5"
BASE_DATA_GEN_PATH = "data/gen/"
FEATURE_EXTRACTION_SRC_PATH = "data/additional_escape_variants/gen"
FEATURE_EXTRACTION_SAVE_PATH = "data/additional_escape_variants/gen"
INTEGRATED_MODEL_PATH = "model/integrated_model_1024"
BASE_ANALYSIS_PATH = "data/analysis/integrated_1024"




#LStm BASED FEATURES
MODEL_LSTM_OUTPUT_FEATURE_ESC_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_features_esc.npz"
MODEL_LSTM_OUTPUT_GISAID_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_output_gisaid_features.npz"
MODEL_LSTM_OUTPUT_GREANY_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_output_greany_features.npz"

from Bio import SeqIO

'''
lengths : array containing sequence length for individual sequence [1152, 1205, ..., 1425]
seq_len : Maximum Length  of sequence (taken into consideration )
returns start index and end index in the form of generator and updates the current index  
'''
def iterate_lengths(lengths, seq_len):

    curr_idx = 0
    for length in lengths:
        if length > seq_len:
            sys.stderr.write(
                'Warning: length {} greather than expected '
                'max length {}\n'.format(length, seq_len)
            )
        yield (curr_idx, curr_idx + length)
        curr_idx += length




def cross_entropy(logprob, n_samples):
    return -logprob / n_samples

def report_performance(model_name, model, vocabulary, train_seqs, test_seqs):
    X_train, lengths_train = model.featurize_seqs(train_seqs, vocabulary)
    logprob = model.score(X_train, lengths_train)
    print('Model {}, train cross entropy: {}'.format(model_name, cross_entropy(logprob, len(lengths_train))))
    X_test, lengths_test = model.featurize_seqs(test_seqs, vocabulary)
    logprob = model.score(X_test, lengths_test)
    print('Model {}, test cross entropy: {}'.format(model_name, cross_entropy(logprob, len(lengths_test))))
def featurize_seqs(seqs):
        vocabulary = FZ.getVocabDictionary()
        start_int = len(vocabulary) + 1
        end_int = len(vocabulary) + 2
        sorted_seqs = sorted(seqs)
        X = np.concatenate([
            np.array([ start_int ] + [
                vocabulary[word] for word in seq
            ] + [ end_int ]) for seq in sorted_seqs
        ]).reshape(-1, 1)
        lens = np.array([ len(seq) + 2 for seq in sorted_seqs ])
        assert(sum(lens) == X.shape[0])
        return X, lens
def batch_train(model, seqs,  batch_size=512,  n_epochs=1):
    # Control epochs here.
    model.n_epochs_ = n_epochs
    n_batches = math.ceil(len(seqs) / float(batch_size))
    print('Traning seq batch size: {}, N batches: {}'.format(batch_size, n_batches))
    for epoch in range(n_epochs):
        random.shuffle(seqs)
        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch =  seqs[start:end] 
            X, lengths = featurize_seqs(seqs_batch)
            model.fit(X, lengths)
            del seqs_batch

def train_embedding_model(seqs, epochs=1):
    seq_len = 23
    vocab_size = 27
    model = get_model_structure(seq_len, vocab_size) ##63 
    
    batch_train(model, seqs, batch_size=32, n_epochs=epochs)
   
      
def get_model_structure(seq_len=23, vocab_size=27):
        from BiLSTMLanguageModeler import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            n_epochs=1,
            batch_size=24,
            inference_batch_size=24,
            cache_dir='embedding_full_model/',
            seed=41,
            verbose=True,
        )
        model.model_.summary()
        return model                  
'''
seq_of_interest - Any single sequence under observation example : wild sequence 
'''
def predict_sequence_prob(seq_of_interest, model):
   
    X_cat, lengths = featurize_seqs(seq_of_interest)
    y_pred = model.predict(X_cat, lengths)
    print("Original y_pred shape: ",y_pred.shape)

    y_reshaped = np.reshape(y_pred, (-1, 22*28))
    print("Original y_pred shape: ",y_reshaped.shape)

    return y_reshaped




def load_fz_model():
    model = get_model_structure()
    model.model_.load_weights(MODEL_PATH)
    return model

def get_features(windowed_file):
    model =load_fz_model()
    seqs = ASE.read_combined_window_file(windowed_file)

    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    return y_embed_output



def get_balanced_datasets_over_sampling(non_sig_train, non_sig_test, sig_train_orig, sig_test):
    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 40166 #366,533 + 40166
    #NOn Sig train test features
    non_sig_train = np.reshape(non_sig_train, (-1, 1024))
    non_sig_train_len = non_sig_train.shape[0]
    non_sig_features_train_output = np.zeros( (non_sig_train_len, 1))
    print(f" non_sig_train shape: {non_sig_train.shape} Non Sig test shape {non_sig_test.shape} ")
    non_sig_train_dataset = tf.data.Dataset.from_tensor_slices((non_sig_train, non_sig_features_train_output))

    non_sig_test = np.reshape(non_sig_test, (-1, 1024))
    non_sig_test_len = non_sig_test.shape[0]
    non_sig_test_features_output = np.zeros( (non_sig_test_len, 1))
    non_sig_test_dataset = tf.data.Dataset.from_tensor_slices((non_sig_test, non_sig_test_features_output))
    

    #Over sample sig train features 
    sig_train_orig = np.reshape(sig_train_orig, (-1, 1024))
    sig_train_features = over_sample_array(sig_train_orig, total_samples = non_sig_train_len)
    sig_train_len =  sig_train_features.shape[0]
    print("Sig Train shape: ",sig_train_features.shape )
    sig_train_features_output = np.ones( (sig_train_len, 1) )
    sig_train_dataset = tf.data.Dataset.from_tensor_slices((sig_train_features, sig_train_features_output))

    sig_test = np.reshape(sig_test, (-1, 1024))
    print("Sig Test shape: ",sig_test.shape)
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

def over_sample_array(arr, total_samples=0):
    #replace True oversamples data
    over_samples = arr[np.random.choice(len(arr), size=total_samples, replace=True)]
    print("Oversamples shape is: ",over_samples.shape)
    return over_samples 

def get_dense_model():
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense 
    from tensorflow.keras.layers import BatchNormalization 
    INPUT_FEATURE_SIZE = 1024
    model = Sequential()
    model.add(layers.Input(INPUT_FEATURE_SIZE, ))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    optmz = tf.keras.optimizers.Adam(
        name = "Adam",
        learning_rate = 0.00001
    )
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optmz, metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()
    print(model.optimizer.get_config())
    
    return model

def analyze_greany():
    
    model = m.load_model(INTEGRATED_MODEL_PATH)
    model.summary()
    sig_data = get_features("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv")
    sig_data = np.reshape(sig_data, (-1, 1024))

    non_sig_data =  get_features("data/additional_escape_variants/gen/greany_non_sig_combined_windowed_seqs.csv")
    non_sig_data = np.reshape(non_sig_data, (-1, 1024))

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    print("Evaluating  greany Combined ")
    model.evaluate(features, targets)
    
    #print("Evaluating  greany Sig only ")
    #model.evaluate(sig_data, sig_features_output)
    #print("Evaluating  greany Non Sig only ")
    #model.evaluate(non_sig_data, nonsig_features_output)

    #Analyze greany predictions | plot AUC_ROC / Confusion Matrix / Precision Recall
    y_preds = model.predict(features)
    plot_greany_test_predictions(targets, y_preds)

def evaluate_baum():
    model = m.load_model(INTEGRATED_MODEL_PATH)
    sig_data = get_features("data/additional_escape_variants/gen/baum_sig_combined_windowed_seqs.csv")
    sig_data = np.reshape(sig_data, (-1, 1024))


    non_sig_data = get_features("data/additional_escape_variants/gen/baum_non_sig_combined_windowed_seqs.csv")
    non_sig_data = np.reshape(non_sig_data, (-1, 1024))

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    #Evaluating greany fetures 
    print("Evaluating  Baum Combined ")
    model.evaluate(features, targets)

    print("Evaluating  Baum Sig only ")
    model.evaluate(sig_data, sig_features_output)

    print("Evaluating  Baum Non Sig only ")
    model.evaluate(non_sig_data, nonsig_features_output)

def execute_aggregate_network():
    non_sig_train  = get_features("data/additional_escape_variants/gen/non_sig_combined_windows_train.csv")
    non_sig_test  = get_features("data/additional_escape_variants/gen/non_sig_combined_windows_test.csv")

    sig_train_orig  = get_features("data/additional_escape_variants/gen/sig_combined_windows_train.csv")
    sig_test =  get_features("data/additional_escape_variants/gen/sig_combined_windows_test.csv")

    train_ds, val_ds = get_balanced_datasets_over_sampling(non_sig_train, non_sig_test, sig_train_orig, sig_test)
    model  = get_dense_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  
    model.fit(train_ds,  epochs=30, validation_data=val_ds,  callbacks= [callback], verbose=1) #verbose 1 = Progress bar
    
    model.save(INTEGRATED_MODEL_PATH)

    

def analyze_validation_dataset():
    model = m.load_model(INTEGRATED_MODEL_PATH)
    model.summary()

    non_sig_data  = get_features("data/additional_escape_variants/gen/non_sig_combined_windows_test.csv")
    non_sig_data = np.reshape(non_sig_data, (-1, 1024))
    
    sig_data =  get_features("data/additional_escape_variants/gen/sig_combined_windows_test.csv")
    sig_data = np.reshape(sig_data, (-1, 1024))

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    print("Evaluating  Validation dataset: ")
    model.evaluate(features, targets)

    #Analyze  predictions | plot AUC_ROC / Confusion Matrix / Precision Recall
    y_preds = model.predict(features)

    plot_validation_predictions(targets, y_preds)


def plot_greany_test_predictions(targets, y_pred):
    model = m.load_model(INTEGRATED_MODEL_PATH)
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_AUC.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_pr_curve.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_confusion_matrix.png')

    print("Greany visual analysis plotted successfully !!!")

def plot_validation_predictions(targets, y_pred):
    model = m.load_model(INTEGRATED_MODEL_PATH)
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/validation_AUC.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/validation_pr_curve.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/validation_confusion_matrix.png')

    print("Validation visual analysis plotted successfully !!!")


def save_embed_learning_features():
    fl_path =  'data/gen/windowed_embed_train_seqs_35527.csv'
    df = pd.read_csv(fl_path)
    seqs = df['window_seqs'].to_list()

    model =load_fz_model()
    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)


    print("Shape of generated embed_learning_features: ", y_embed_output)
    # save to npy file
    np.save('data/analysis/embed_learning_feature/embed_learned_features.npy', y_embed_output)

def analyze_embed_learning_features_umap():
    features  = np.load('data/analysis/embed_learning_feature/embed_learned_features.npy')
    print("Shape of embed learned features: ", features.shape)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='SARS-COV-2 Escape Prediction')
    parser.add_argument('--predict', type=str, default='none',
                        help='Predict Greaney Sequences')
    parser.add_argument('--escape', type=str, default='' )
    parser.add_argument('--position', type=int )
    args = parser.parse_args()
    return args

def predict_new_sequence(sequence, position):
    model = m.load_model(INTEGRATED_MODEL_PATH)




if __name__ == "__main__":
    #seqs_train =  ASE.read_window_file('data/gen/windowed_embed_train_seqs_35527.csv')
    #train_embedding_model(seqs_train, epochs=4)
    execute_aggregate_network()
    analyze_greany()
    evaluate_baum()
    analyze_validation_dataset()

    



    




import sys
import os

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    concatenate, Activation, Dense, Embedding, LSTM, Reshape)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
from transformers import FEATURE_EXTRACTOR_MAPPING
np.random.seed(1)
import math
import pandas as pd

import AdditionalSignficantEscapeSeqGenerator as ASE
import featurizer as FZ
MODEL_PATH = "fe_model_target/checkpoints/bilstm/bilstm_1024-01.hdf5"
BASE_DATA_GEN_PATH = "data/gen/"
FEATURE_EXTRACTION_SRC_PATH = "data/additional_escape_variants/gen"
FEATURE_EXTRACTION_SAVE_PATH = "data/additional_escape_variants/gen"

#Model OUTPUT Based Features
MODEL_OUTPUT_FEATURE_ESC_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_output_features_esc.npz"
MODEL_OUTPUT_GISAID_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_output_gisaid_features.npz"
MODEL_OUTPUT_GREANY_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_output_greany_features.npz"

#LStm BASED FEATURES
MODEL_LSTM_OUTPUT_FEATURE_ESC_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_features_esc.npz"
MODEL_LSTM_OUTPUT_GISAID_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_output_gisaid_features.npz"
MODEL_LSTM_OUTPUT_GREANY_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_output_greany_features.npz"



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
def batch_train(model, seqs,  batch_size=50,  n_epochs=1):
    # Control epochs here.
    model.n_epochs_ = n_epochs
    n_batches = math.ceil(len(seqs) / float(batch_size))
    print('Traing seq batch size: {}, N batches: {}'.format(batch_size, n_batches))
    for epoch in range(n_epochs):
        random.shuffle(seqs)
        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch =  seqs[start:end] 
            X, lengths = featurize_seqs(seqs_batch)
            model.fit(X, lengths)
            del seqs_batch


def train_embedding_model(seqs):
    seq_len = 23
    vocab_size = 27
    model = get_model_structure(seq_len, vocab_size) ##63 
    
    batch_train(model, seqs, batch_size=64, n_epochs=1)
   
      
def get_model_structure(seq_len=23, vocab_size=27):
        from BiLSTMLanguageModeler import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=1024,
            n_hidden=2,
            n_epochs=1,
            batch_size=10,
            inference_batch_size=2048,
            cache_dir='fe_model_target/',
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
def load_model():
    model = get_model_structure()
    model.model_.load_weights(MODEL_PATH)
    return model


'''
This method provides the output features for the embedding training network
input : The input n sequences for which we would like to computer feature vectors
Output: (n*22, 28) dimension array
'''
def get_model_output_features(pred_seq_path):
    seqs =  ASE.read_window_file(pred_seq_path)
   
    #As we have input of 22 Length for one seq : ouput will be of 22 Lenght with 28 dimension representation for each  
    model = load_model() 
    y_pred = predict_sequence_prob(seqs, model)
    print("Prediction shape: ", y_pred.shape)
    return y_pred


def get_model_lstm_output_feature(input_seqs_path):
    print("*Getting embed features using LSTM model output features")
    seqs =  ASE.read_window_file(input_seqs_path)
    model = load_model() 

    X_cat, lengths = featurize_seqs(seqs)

    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape)
    y_embed_output_reshaped = np.reshape(y_embed_output, (-1, 45056) ) #y_embed_output.reshape( ) #22*2048
    print("y_embed_output predictions shape: ", y_embed_output_reshaped.shape)
    return y_embed_output_reshaped


def save_features(sig_features, nonsig_features, save_path):
    np.savez(save_path, SIG = sig_features,  NON_SIG =  nonsig_features )
    print("Sig and Non sig features are saved to path :", save_path)

def get_features(save_path):
    features = np.load(save_path)
    sig_features = features['SIG']
    non_sig_features = features['NON_SIG']

    print(f"Retrieved shapes: SIG: {sig_features.shape} Non SIG: {non_sig_features.shape}")
    
    return (sig_features, non_sig_features)




'''
These features extracted are based on the ouput of the embedding trained network
'''
def generate_model_output_features(sig_file, non_sig_file):
    sig_features  = get_model_output_features(sig_file)
    non_sig_features  = get_model_output_features(non_sig_file)
    save_features(sig_features, non_sig_features, MODEL_OUTPUT_FEATURE_ESC_SAVE_PATH)

def generate_model_gisaid_output_features(sig_file, non_sig_file, output_file ):
    sig_features  = get_model_output_features(sig_file)
    non_sig_features  = get_model_output_features(non_sig_file)
    total_len = len(sig_features) + len(non_sig_features)

    save_features(sig_features, non_sig_features, output_file)

'''
Significant Mutant information is taken from ESC resource
Non Significant Information is choosen from Greany 
We will use greany non significant for test set 
'''
def generate_model_lstm_esc_output_features():
    esc_sig_file = FEATURE_EXTRACTION_SRC_PATH +  "/sig_train_mut_window_size_20.csv"
    greany_non_sig_file = FEATURE_EXTRACTION_SRC_PATH + "/greany_not_sig_mut_window_size_20.csv"
    save_path  =  FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_esc_features.npz"

    lstm_based_sig_features = get_model_lstm_output_feature(esc_sig_file)
    lstm_based_non_sig_features = get_model_lstm_output_feature(greany_non_sig_file)

    save_features(lstm_based_sig_features, lstm_based_non_sig_features, save_path)

def generate_model_lstm_gisaid_output_features(sig_file, non_sig_file, output_file):
    lstm_based_sig_features = get_model_lstm_output_feature(sig_file)
    lstm_based_non_sig_features = get_model_lstm_output_feature(non_sig_file)
    save_features(lstm_based_sig_features, lstm_based_non_sig_features, output_file)

def generate_model_lstm_combined_output_features(input_file_path, output_file_name):
    print(f"Getting embed features using LSTM model output features for input file {input_file_path}")
    seqs = ASE.read_combined_window_file(input_file_path)
    model = load_model() 
    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape)
    y_embed_output_reshaped_data = np.reshape(y_embed_output, (-1, 45056) ) #y_embed_output.reshape( ) #22*2048
    
    save_path = FEATURE_EXTRACTION_SAVE_PATH + os.path.sep + output_file_name
    #np.savez(save_path,  y_embed_output_reshaped)
    save_to_hdf5(save_path, y_embed_output_reshaped_data)

    print("y_embed_output predictions file with shape: ", y_embed_output_reshaped_data.shape , " is saved  to path: ",save_path)

'''
Implemented specifically After  Baum 
'''
def generate_model_lstm_combined_output_features_feature_reduced(input_file_path, output_file_name):
    print(f"Getting embed features using LSTM model output features for input file {input_file_path}")
    seqs = ASE.read_combined_window_file(input_file_path)
    model = load_model() 
    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 2048)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    
    save_path = FEATURE_EXTRACTION_SAVE_PATH + os.path.sep + output_file_name
    np.savez(save_path,  y_embed_output)

    print("y_embed_output predictions file with shape: ", y_embed_output.shape , " is saved  to path: ",save_path)

def save_to_hdf5(file_path, data):
    import h5py
    #Write data to h5
    hf = h5py.File(file_path, 'w')
    hf.create_dataset('non_sig_seqs', data=data)
    print("Data saved successfully to Hdf5")
    hf.close()


def generate_greany_model_output_features():
    greany_src_file = FEATURE_EXTRACTION_SRC_PATH +  "/greany_sig_mut_window_size_20.csv"
    sig_features  = get_model_output_features(greany_src_file)
    np.savez(MODEL_OUTPUT_GREANY_FEATURE_SAVE_PATH, SIG = sig_features)
    print("Greany Model output feature are save to path :", MODEL_OUTPUT_GREANY_FEATURE_SAVE_PATH)

def generate_greany_model_lstm_output_features():
    greany_src_file = FEATURE_EXTRACTION_SRC_PATH +  "/greany_sig_mut_window_size_20.csv"
    sig_features  = get_model_lstm_output_feature(greany_src_file)
    np.savez(MODEL_LSTM_OUTPUT_GREANY_FEATURE_SAVE_PATH, SIG = sig_features)
    print("Greany Model output feature are save to path :", MODEL_LSTM_OUTPUT_GREANY_FEATURE_SAVE_PATH)

def get_greany_model_output_features():
    features = np.load(MODEL_OUTPUT_GREANY_FEATURE_SAVE_PATH)
    sig_features = features['SIG']
    print(f"Retrieved shapes: Greany SIG: {sig_features.shape} ")

    return sig_features

def get_greany_model_lstm_output_features():
    features = np.load(MODEL_LSTM_OUTPUT_GREANY_FEATURE_SAVE_PATH)
    sig_features = features['SIG']
    print(f"Retrieved shapes: Lstm Greany SIG: {sig_features.shape} ")

    return sig_features



if __name__ == "__main__":

    #seqs_train =  ASE.read_window_file('data/gen/windowed_seqs_20_length_windows_310421.csv') ##63 
    #train_embedding_model(seqs_train)


    #Esc combined features 
    #generate_model_lstm_combined_output_features("data/additional_escape_variants/gen/esc_sig_combined_windowed_seqs.csv", "esc_sig_combined_fz.npz")
    
    #Gsiad combined fetures (First feature wild type | Next feature Mutated type)
    generate_model_lstm_combined_output_features_feature_reduced('data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_181887.csv', 'reduced_non_sig_features_181887.npz')
    #generate_model_lstm_combined_output_features_feature_reduced('data/additional_escape_variants/gen/sig_combined_windows.csv', 'sig_combined_windows_fz_reduced_new.npz')

    #Greany combined features
    #generate_model_lstm_combined_output_features("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv", "greany_sig_combined_fz.npz")
    #generate_model_lstm_combined_output_features("data/additional_escape_variants/gen/greany_non_sig_combined_windowed_seqs.csv", "greany_non_sig_combined_fz.npz")

    #Baum Features 
    #generate_model_lstm_combined_output_features_feature_reduced('data/additional_escape_variants/gen/baum_sig_combined_windowed_seqs.csv', 'buam_sig_combined_fz.npz')
    #generate_model_lstm_combined_output_features_feature_reduced('data/additional_escape_variants/gen/baum_non_sig_combined_windowed_seqs.csv', 'buam_non_sig_combined_fz.npz')

   

    

    pass



    




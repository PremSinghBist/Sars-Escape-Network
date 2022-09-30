import numpy as np 
import escape_validation as EV
import tensorflow as tf

def get_test_data(test_data_path):
    gram, change =  EV.read_analysis_output(test_data_path) 

    #Converting to numpy array with N*1 column shape 
    np_change =  np.array([change]).reshape(-1, 1) 
    np_gram = np.array([gram]).reshape(-1, 1)

    features = np.concatenate( (np_change, np_gram), axis=1)

    features = remove_row_with_nan(features)
    print(f"Test file: {test_data_path} after removing Nan rows shape: {features.shape}")

    return features


def get_escape_featured_data():
    non_sig_gram, non_sig_change =  EV.read_analysis_output('gsaid_non_significant_complete__analysis_17000Seq.txt') 
    
    sig_gram, sig_change = EV.read_analysis_output('GISAID_SIG_SAMPLES_output.txt')
    print(f"non_sig_gram: {len(non_sig_gram)} non_sig_change :  {len(non_sig_change)} ")
    print(f"sig_gram: {len(sig_gram)} sig_change :  {len(sig_change)} ")

    #Converting to numpy array with N*1 column shape 
    np_sig_change =  np.array([sig_change]).reshape(-1, 1) 
    np_sig_gram = np.array([sig_gram]).reshape(-1, 1)
    sig_output = np.ones( (len(np_sig_gram) , 1)) 

    np_non_sig_change = np.array([non_sig_gram]).reshape(-1, 1)
    np_non_sig_gram = np.array([non_sig_gram]).reshape(-1, 1) 
    non_sig_output = np.zeros( (len(np_non_sig_gram) , 1))

   
    
    #Get combined features of semntic change and grammar | merging features across columns | col1: change col2: grammar
    sig_feautres = np.concatenate( (np_sig_change, np_sig_gram, sig_output ), axis=1)
    non_sig_features = np.concatenate( (np_non_sig_change, np_non_sig_gram, non_sig_output ), axis=1)


    print(f"sig_feautres shape: {sig_feautres.shape} non_sig_features shape: {non_sig_features.shape}")

    #Combined features first n rows for sig | next other rows for non sig 
    features  = np.concatenate( (sig_feautres, non_sig_features) , axis = 0) 
    print(f"Combined features shape: {features.shape} ")

    features = remove_row_with_nan(features)
    print(f"After removing Nan rows: {features.shape}")


     #Shuffle data 
    np.random.shuffle(features)
    #print(features[0: 5])

    return features
'''
Removes a row that contains Nan value
'''
def remove_row_with_nan(features):
    #Get the indexes of all 
    isNan_bool_array = np.isnan(features).any(axis=1) #isNan_bool_array - Provides array of True  if any columns contains Nan value else returns false 
    features = np.delete(features, isNan_bool_array, axis = 0) #Deletes entire row if its contains Nan value
    return features





def evaluate_greany_datasets(model):
    #evaluate greanydataset 
    greany_sig_path =  'greany_science_0.3.txt' #"/home/perm/viral-mutation-master/results/cov/unique_significant_sequence_result/greany_science_0.3.txt"
    greany_sig_data = get_test_data(greany_sig_path)
    greany_sig_targets  = np.ones( (len(greany_sig_data) , 1)) 
    LanguageModelFE.evaluate_model(model, greany_sig_data, greany_sig_targets, 'Greany Sig Seqs')

    greany_non_sig_path = 'greany_science_not_sig_30_samples.txt'
    greany_non_sig_data = get_test_data(greany_non_sig_path)
    greany_non_sig_targets  = np.zeros( (len(greany_non_sig_data) , 1))
    LanguageModelFE.evaluate_model(model, greany_non_sig_data, greany_non_sig_targets, 'Greany Non Sig Seqs')

def get_training_data(get_escape_featured_data):
    featured_data  = get_escape_featured_data()

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(featured_data, test_size=0.33, random_state=42)
    print(f'Shape of train_data : {train_data.shape} Test Shape: {test_data.shape}')

    train_x = train_data[: , 0:2]
    train_y = train_data[:, -1]

    test_x = test_data[: , 0:2]
    test_y = test_data[:, -1]

    print("train_data shape",  train_x.shape )
    return train_x,train_y,test_x,test_y

def train_evaluate_model():
    train_x, train_y, test_x, test_y = get_training_data(get_escape_featured_data) 
    model = create_baseline_model()
    history = model.fit(train_x, train_y, batch_size=32, epochs=2,
                        validation_data=(test_x, test_y), verbose=1)

    evaluate_greany_datasets(model)
    model.save('lang_fe/fe_model')

if __name__ == "__main__":
    train_evaluate_model()

    
  
    
 

    















    

    
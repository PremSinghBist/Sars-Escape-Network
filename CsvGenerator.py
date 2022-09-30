from escape import BAUM_SEQ_PATH
import os 
SEP=os.path.sep

import pandas as pd 
def generate_csv_using_dict(dict, file_name, header=True):
    dataFrame = pd.DataFrame.from_dict(dict, orient='index') #index means key should be row | orient='column' means : keys = columns
    dataFrame.to_csv(file_name, header= header)

def generate_grouped_baum_seqs(dict, SEQS_LIMIT):
    grouped_dict = {
    }
    for seq, mutant in dict.items():
        if mutant in grouped_dict:
            seqs = grouped_dict[mutant]
            if len(seqs) < SEQS_LIMIT:
                 seqs.append(seq)
        else:
            grouped_dict[mutant] = [seq]

    return grouped_dict

def write_grouped_baum_seqs(dict, file_name):
    all_mutants = []
    all_seqs = []
    for mutant, seq in dict.items():
        seqs = dict[mutant]
        for s in seqs:
            all_seqs.append(s)
            all_mutants.append(mutant)
    import numpy as np
    dataFrame  = pd.DataFrame( np.transpose([all_mutants, all_seqs]) )
    dataFrame.to_csv(file_name, index=False, header=None) #Index False not print 1, 2, 3 index  | header=None will not print header
    print("Grouped Baum Data Written successfully. ")

def generate_grouped_baum_seqs_wrapper(src_file, dest_file, SEQS_LIMIT):
    baum_df = read_csv(src_file)
    baum_dict = {}
    for index, row in baum_df.iterrows():
        baum_dict[row[0]] = row[1]
    
    grouped_dict = generate_grouped_baum_seqs(baum_dict, SEQS_LIMIT)
    write_grouped_baum_seqs(grouped_dict, dest_file)


'''
Returns a pandas dataframe for csv file 
'''
def read_csv(file_name, header=None):
    df = pd.read_csv(file_name, header=header, engine='python')
    return df

def write_csv(data_dict = '', file_path = ''):
    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    # src_file = BAUM_SEQ_PATH + SEP +"baum.csv"
    # dest_file =  BAUM_SEQ_PATH + SEP +"baum_grouped_3.csv"
    # generate_grouped_baum_seqs_wrapper(src_file, dest_file , 3)
    pass
    

import sys
import re

from torch import equal
from AA_change_patterns import single_residue_subsitution
from MutationTypes import MutationTypes
from escape import GREANY_NON_SIG_MUT_FILE, BAUM_SIG_MUT_FILE, BAUM_NON_SIG_MUT_FILE, readAdditionalMutants
from escape import read_wildSequence
from escape import get_baum_mutants

from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import SignficantEscapeGenerator as SEG
from escape import ADDITIONAL_MUTANT_GENERATED_PATH 
from escape import BAUM_SEQ_PATH
from escape import GREANY_SIG_MUT_FILE
from escape import GREANY_SEQ_PATH
import pandas as pd
import CsvGenerator 
import os
import numpy as np
import featurizer as FZ
import pandas as pd
import h5py


np.random.seed(1)
SIG_INDEX = "SIG"
NON_SIG_INDEX = "NON_SIG"
GSAID_SIG_SPIKE_PATH = "/home/perm/cov/data/gen/GISAID_SIG.fa"
SINGLE_RES_TRAIN_MUT_WITHOUT_GREANY = 'single_res_sig_train_mut_without_greany.csv'
GISAID_NON_SIG_COMBINED_WINDOWED_SEQS_SHARDED_PATH = 'data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_sharded.h5'
TOTAL_NO_OF_SHARDS = 72

FEATURE_EXTRACTION_SRC_PATH = "data/additional_escape_variants/gen"

FEATURE_ESC_SIG_COMBINED_PATH = FEATURE_EXTRACTION_SRC_PATH + "/esc_sig_combined_fz.npz"
FEATURE_GISAID_SIG_COMBINED_PATH = FEATURE_EXTRACTION_SRC_PATH + "/gisaid_sig_combined_fz.npz"
FEATURE_GISAID_NON_SIG_COMBINED_PATH = FEATURE_EXTRACTION_SRC_PATH + "/gisaid_non_sig_combined_fz.npz"

#greany 
FEATURE_GREANY_SIG_COMBINED_PATH = FEATURE_EXTRACTION_SRC_PATH + "/greany_sig_combined_fz.npz"
FEATURE_GREANY_NON_SIG_COMBINED_PATH = FEATURE_EXTRACTION_SRC_PATH + "/greany_non_sig_combined_fz.npz"


#Reduces feature save Path 
# "/reduced_non_sig_features_153140.npz"    #"/reduced_non_sig_features.npz" 
REDUCED_NONSIG_FEATURE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/reduced_non_sig_features_181887.npz"
REDUCED_NONSIG_LARGE_DATA_FEATURE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/gisaid_non_sig_combined_fz_363744_reducedfz.npz"
REDUCED_SIG_FEATURE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/reduced_sig_features.npz"
REDUCED_SIG_FEATURE_NEW_PATH = FEATURE_EXTRACTION_SRC_PATH +'/sig_combined_windows_fz_reduced_new.npz' #This actually removes greany if any present
REDUCED_GREANY_FEATURE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/reduced_greany_features.npz"
#Shared instance for reading non sig shared gisaid h5 dataset


#Baum 
REDUCED_BAUM_SIG_FEATURE_PATH =  FEATURE_EXTRACTION_SRC_PATH + "/buam_sig_combined_fz.npz"
REDUCED_BAUM_NON_SIG_FEATURE_PATH =  FEATURE_EXTRACTION_SRC_PATH + "/buam_non_sig_combined_fz.npz"


hf = None


import math
import random

import CsvGenerator as CSV_UTIL

SEQS = {
       MutationTypes.SINGLE_RES_SUB : [],
       MutationTypes.MULTIPLE_RES_SUB: [],
       #Delete
       MutationTypes.DEL_SINGLE_RES: [],
       MutationTypes.DEL_RES_1_OR_2: [],
       MutationTypes.DEL_IN_RANGE: [],
       #Frequent Delete
       MutationTypes.DEL_SINGLE_RES_FREQ: [],
       MutationTypes.DEL_IN_RANGE_FREQ: [],
       #Insert Inbetween
       MutationTypes.INSERT_IN_BETWEEN: []
}


def construct_single_residue_mut_seqs(wildSeq):
    single_residue_subsitution_mutations = readAdditionalMutants(MutationTypes.SINGLE_RES_SUB.name+".csv")
    for mutant in single_residue_subsitution_mutations:
        wildSeq_arry = [wildSeq[i] for i in range(len(wildSeq))]
        #Example E848K - Extract E | 848 | K
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])

        #Change the residue at mutated position | index starts from zero so, -1
        wildSeq_arry[changed_position-1] = changed_to_residue
        mutated_seq = ''.join(wildSeq_arry)
        mutated_seq_info = {mutated_seq : mutant}
        SEQS.get(MutationTypes.SINGLE_RES_SUB).append(mutated_seq_info)
def construct_multiple_residue_mut_seqs(wildSeq):
    multiple_residue_subsitution_mutations = readAdditionalMutants(MutationTypes.MULTIPLE_RES_SUB.name+".csv")
    for mutant in multiple_residue_subsitution_mutations:
        wildSeq_arry = [wildSeq[i] for i in range(len(wildSeq))]
        #Example: KVG444-6TST
        stripped = mutant.split("-")
        first_term = stripped[0] #KVG444
        last_term = stripped[1]  #6TST
        
        original_residues = re.findall(r"[A-Z]+", first_term)[0] #KVG
        position = int(re.findall(r"\d+", first_term)[0]) - 1 #444 -->443 
        changed_residue = re.findall(r"[A-Z]+", last_term)[0] #TST
        assert len(original_residues) ==  len(changed_residue), 'Original residues and Mutated residues length must be same'

        for i in range(len(original_residues)):
            wildSeq_arry[position] = changed_residue[i]
            position = position+1
        
        mutated_seq = ''.join(wildSeq_arry)
        mutated_seq_info = {mutated_seq: mutant}
        SEQS.get(MutationTypes.MULTIPLE_RES_SUB).append(mutated_seq_info)

def construct_single_residue_del_seqs(wildSeq):
    single_residue_del_mutations = readAdditionalMutants(MutationTypes.DEL_SINGLE_RES.name+".csv")
    for mutant in single_residue_del_mutations:
        wildSeq_arry = [wildSeq[i] for i in range(len(wildSeq))]
        #Example: F140del
        deleted_residue = mutant[0]
        deleted_position = int(re.findall(r"\d+", mutant)[0]) - 1 
        print("Deleted residue:{}, Position: {}".format(deleted_residue, deleted_position))

        wildSeq_arry.pop(deleted_position) #Delete residue from particular index
        mutated_seq = ''.join(wildSeq_arry)
        mutated_seq_info = {mutated_seq: mutant}
        SEQS.get(MutationTypes.DEL_SINGLE_RES).append(mutated_seq_info)

def generate_fasta_file(mutationType):
    sequences = SEQS.get(mutationType)
    write_records(sequences , mutationType.name) 
    print("File Generated Successfully for type: ",mutationType.name)

def write_records(sequences , mutationType):
    sequence_records = []
    
    if len(sequences) == 0:
        print("Single residue subsitution Sequence is empty") 
    else:
        sequence_num = 0
        for seq_info in sequences:
            for sequence, mutant_info in seq_info.items():
                record = SeqRecord(
                    Seq(str(sequence),), 
                    id=str(sequence_num) + "|"+mutant_info,
                    description='|'
                    )
                sequence_records.append(record)
            sequence_num = sequence_num+1
    generated_file_name = ADDITIONAL_MUTANT_GENERATED_PATH+"/"+mutationType+".faa"
    print(generated_file_name)
    SeqIO.write(sequence_records, generated_file_name, "fasta") 

def generate_single_residue_mutation(wildSeq):
    construct_single_residue_mut_seqs(wildSeq)
    generate_fasta_file(MutationTypes.SINGLE_RES_SUB)

def generate_multiple_residue_subsitution(wildSeq):
    construct_multiple_residue_mut_seqs(wildSeq)
    generate_fasta_file(MutationTypes.MULTIPLE_RES_SUB)

def generate_single_residue_del_seqs(wildSeq):
    construct_single_residue_del_seqs(wildSeq)
    generate_fasta_file(MutationTypes.DEL_SINGLE_RES)

def getFileFullPath(mutationType):
    return ADDITIONAL_MUTANT_GENERATED_PATH+"/"+mutationType+".faa"

'''
This method will read significant sequences from two datasets and keep unique sequences contained in additionally genrated significant mutations
'''
def generateUniqueSignificantSeqs():
    original_dataset_path = SEG.getSignificantFilePath()
    orig_seqs = SEG.read_sequences(original_dataset_path)
    
    additional_dataset_path = getFileFullPath(MutationTypes.SINGLE_RES_SUB.name) 
    additional_seq =  SEG.read_sequences(additional_dataset_path)
    print("Original Dataset path: {} Additional Dataset path: {} ".format(original_dataset_path, additional_dataset_path ))
    print('Original seq Len: {}, Additional Seq Len {}'.format(len(orig_seqs), len(additional_seq)))

    #if original sequence is also present in additional sequence, then remove from additional sequence
    for seq in orig_seqs:
        if seq in additional_seq:
            index = additional_seq.index(seq)
            additional_seq.pop(index)

    return additional_seq

def generateUniqueNotSignificantSeqs():
    not_significant_seqs = remove_duplicates_from_non_singnificant_seq()
    SEG.generate_fasta_using_list(not_significant_seqs, SEG.DUPLICATE_REMOVED_NOT_SIGNIFICANT_FILE)

    

'''
Check if Not Significant sequences contains significant sequences 
-Non Significant sequences were constructed from original dataset of 2020
-The observations which 
-if so remove significant sequences from Not significnat dataset  
'''
def remove_duplicates_from_non_singnificant_seq():
    not_sig_path = SEG.getNotSignificantFilePath()
    not_significant_seq = SEG.read_sequences(not_sig_path)
    print("***Not Sig: File {} , Len: ;{}".format(not_sig_path, len(not_significant_seq) ))


    additional_dataset_path = getFileFullPath(MutationTypes.SINGLE_RES_SUB.name) 
    significant_additional_seq =  SEG.read_sequences(additional_dataset_path)
    print("***Additional seq File path: {} , Len: {}".format(additional_dataset_path, len(significant_additional_seq) ))
    duplicatesCount = 0
    for seq in not_significant_seq:
        if seq in significant_additional_seq:
            duplicatesCount += 1
            #Remove the signficant seq from Non significant ones 
            duplicateSignificantIndex = not_significant_seq.index(seq)
            not_significant_seq.pop(duplicateSignificantIndex)

    print("Duplicates Count: ", duplicatesCount)
    print("***Not Significant seq Len after removing duplicates: {}".format(len(not_significant_seq) ))
    return not_significant_seq
'''
This method removes significant sequences from the list of sequences 
'''
def remove_singnificant_sequences(mixed_seqs_path):
    mixed_sequences = SEG.read_sequences(mixed_seqs_path)
    print("***Sequence File path {} , Len: ;{}".format(mixed_seqs_path, len(mixed_sequences) ))
    return remove_significant_sequences_from_list(mixed_sequences)

'''
Removes significant seqences from given sequence list
'''
def remove_significant_sequences_from_list(sequences):
    duplicatesCount = 0
    sameLenSeq = 0 
    significant_seqs_dataset_path = getFileFullPath(MutationTypes.SINGLE_RES_SUB.name) 
    significant_seqs =  SEG.read_sequences(significant_seqs_dataset_path)
    print("***Significant seqs File path: {} , Len: {}".format(significant_seqs_dataset_path, len(significant_seqs) ))
    for seq in sequences:
        #Seq contains file some thing like : ABBC* , so we need to remove it from there
        seq = seq.strip("*")
        if seq in significant_seqs:
            duplicatesCount += 1
           
            #Remove the signficant seq from Non significant ones 
            duplicateSignificantIndex = sequences.index(seq)
            sequences.pop(duplicateSignificantIndex)

    print("Duplicates Count: ", duplicatesCount)
    print("***Not Significant seq Len after removing duplicates: {}".format(len(sequences) ))
    return sequences
 
def remove_duplicate_sequences(sequence_list):
    unique_list  = set(sequence_list)
    print("Before: {} Size After duplicate removed: {}".format(len(sequence_list) , len(unique_list)  ))
    return list(unique_list)

'''
This method perform the follwoing things : 
 1. It removes duplicate sequences from gsaid raw sequence source and generate new sequences
 2. From new obtained sequences, it removes significant sequeces if present
 3. Generates fasta file
'''
def construct_gsaid_non_significant_sequences(seq_name):
    seq_path = SEG.getGsaid_raw_seq_path()
    seqs  = SEG.read_sequences(seq_path)
    print("Total Sequences: ",len(seqs))

    unique_seqs = remove_duplicate_sequences(seqs)
    non_significant_seqs = remove_significant_sequences_from_list(unique_seqs)
    SEG.generate_fasta_using_list(non_significant_seqs, seq_name)
'''
Splits sequenences into signficant and non significant sequences 
returns sequences, significant_indexes, non_significant_indexes
'''
def split_gsaid_sequences():
    seq_path = SEG.getGsaid_raw_seq_path()

    seqs  = SEG.read_sequences(seq_path)
    print("Total Sequences: ",len(seqs))
    unique_seqs = remove_duplicate_sequences(seqs)
    sequenceTypes = identify_sequence_type(unique_seqs)
    significant_indexes = sequenceTypes[SIG_INDEX] 
    non_significant_indexes = sequenceTypes[NON_SIG_INDEX] 
    print(f"Significant indexes len: {len(significant_indexes)} Non Sig: {len(non_significant_indexes)}")
    return unique_seqs, significant_indexes, non_significant_indexes

'''
This method generates GISAID significant and non significant sequences based with the name provided in param
@param - sig_seq_name : Name of significant sequence file
@param - non_sig_seqs_name : Name of non significnat sequence file to be generated
'''
def generate_gsaid_sequences(sig_seq_name, non_sig_seqs_name):
    seqs, significant_indexes, non_significant_indexes = split_gsaid_sequences()
    sig_seqs =  [seqs[index_value] for index_value in significant_indexes ]
    non_sig_seqs = [seqs[index_value] for index_value in non_significant_indexes]

    print(f"Lenght of GISAID Sig Seq: {len(sig_seqs)} Non Sig {len(non_sig_seqs)}")
    SEG.generate_fasta_using_list(sig_seqs, sig_seq_name)
    SEG.generate_fasta_using_list(non_sig_seqs, non_sig_seqs_name)
    print("GISAID Sequences Generated Successfully!!!")

def generate_gsaid_significant_samples(significant_seqs_samples, sample_name='GISAID_SIG_SAMPLES.fa'):
    SEG.generate_fasta_using_list(significant_seqs_samples, sample_name)
    print(f"GISAID sample Sequences Generated Successfully!!! Total samples : {len(significant_seqs_samples)}")

    

'''
This method identies whether the sequence is significiant or not significant 
and retrives their sequence index positions as dictionary
'''
def identify_sequence_type(sequences):
    significant_indexes = []
    non_significant_indexes = []
    for index, seq in enumerate(sequences):
        seq = cleanse_gsaid_seq(seq)
        sequences[index] = seq
        if is_significant_sequence(seq):
            significant_indexes.append(index)
        else:
            non_significant_indexes.append(index)
   
    return {SIG_INDEX: significant_indexes, NON_SIG_INDEX: non_significant_indexes}
'''
GSAID sequence contains stop codon as * 
we remove * from the sequence and remove space in both side if present.
'''
def cleanse_gsaid_seq(seq):
    #It will remove the trailing stop codon * from the sequence
   return seq.strip("*")

    


def is_significant_sequence(seq):
    if is_single_residue_significant_seq(seq):
        return True
    if is_multiple_residue_significant_seq(seq):
        return True
    if is_delete_residue_significant_seq(seq):
        return True
    if is_insert_residue_significant_seq(seq):
        return True
    else:
        return False
'''
Given an input sequence, it finds if the sequence belongs single residue mutation signficant sequence
returns True if it belongs to significant sequence
'''    
def is_single_residue_significant_seq(seq):
    mutants = readAdditionalMutants(MutationTypes.SINGLE_RES_SUB.name+".csv")
    for mutant in mutants:
        #Example E848K - Extract E | 848 | K
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        if len(seq)-1 < changed_position :
            #print("Observed sequence is shorter in length than mutant position! so discarding it.")
            continue

        if seq[changed_position] == changed_to_residue.strip():
            return True
    return False
''''
-Mutant Set generated from NLP preidcts escape
This method check if a sequence contains a position that is found in greany signficant site
'''
def is_greany_significant_mutant(seq):
    mutants = readAdditionalMutants("greany_significant_mutants.csv")
    #print(f'Greany Significant Mutants List: {len(mutants)}')
    for mutant in mutants:
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        if len(seq)-1 < changed_position :
            #print("Observed sequence is shorter in length than mutant position! so discarding it.")
            continue

        if seq[changed_position] == changed_to_residue.strip():
            return True
    return False

def is_baum_significant_seq(seq):
    muts = get_baum_mutants()
    for mutant in muts:
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        if len(seq)-1 < changed_position :
            #print("Observed sequence is shorter in length than mutant position! so discarding it.")
            continue

        if seq[changed_position] == changed_to_residue.strip():
            return (True, mutant)
    return (False, -1)

def identify_baum_sequences():
    #Stores key = seq, value = Position
    baums_seqs = {   }
    seqs  = SEG.read_sequences(GSAID_SIG_SPIKE_PATH)
    SEQ_LIMIT = 500
    limit_count = 0
    for seq in seqs:
        is_sig, mutant = is_baum_significant_seq(seq)
        if(is_sig == True ):
            limit_count = limit_count +1
            baums_seqs[seq] = mutant
        if limit_count > SEQ_LIMIT:
            return baums_seqs

    return baums_seqs
def generate_baum_sequences(file_name):
    baums_seqs  = identify_baum_sequences()
    file_name = BAUM_SEQ_PATH+ os.path.sep + file_name
    CsvGenerator.generate_csv_using_dict(baums_seqs, file_name, header=False)
    print("Baum Sequences Generated Successfully !!!")





'''
Given sequence: KVG444-6TST
if for a given sequence the residues ranging from 444 to 446 residues are replaced by TST 
then the given sequence is significant
'''
def is_multiple_residue_significant_seq(seq):
    mutants = readAdditionalMutants(MutationTypes.MULTIPLE_RES_SUB.name+".csv")
    for mutant in mutants:
        #Example: KVG444-6TST
        stripped = mutant.split("-")
        first_term = stripped[0] #KVG444
        last_term = stripped[1]  #6TST
        original_residues = re.findall(r"[A-Z]+", first_term)[0] #KVG
        position = int(re.findall(r"\d+", first_term)[0]) - 1 #444 -->443 
        changed_residue = re.findall(r"[A-Z]+", last_term)[0] #TST

        if len(seq)-1 < position + len(changed_residue):
            continue
        if seq[position : position+ len(changed_residue)] ==changed_residue:
            return True
    return False

def is_delete_residue_significant_seq(seq):
    mutants = readAdditionalMutants(MutationTypes.DEL_SINGLE_RES.name+".csv")
    for mutant in mutants:
        #Example: F140del #If reside F at 140 position is absent then it means mutation
        deleted_residue = mutant[0]
        position = int(re.findall(r"\d+", mutant)[0]) - 1 
        if len(seq)-1 < position:
            continue
        if seq[position] != deleted_residue:
            return True

    return False
def is_insert_residue_significant_seq(seq):
    #Mutatnt: 248aKTRNKSTSRRE248k
    startPosition = 247
    insert_mutant = "AKTRNKSTSRREK"
    if(seq[startPosition: startPosition + len(insert_mutant)] == insert_mutant):
        return True
    return False

'''
Given an input sequence, it finds if the sequence contains significant mutants at specific position
returns the mutant dictionary with mutated position information  
'''    
def get_mutant_info_from_seq(seq):
    mut_dic = {}
    single_sub_muts = readAdditionalMutants(MutationTypes.SINGLE_RES_SUB.name+".csv")
    mul_sub_mutants = readAdditionalMutants(MutationTypes.MULTIPLE_RES_SUB.name+".csv")
    #For single res mutant if found
    for mutant in single_sub_muts:
        #Example E848K - Extract E | 848 | K
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        if len(seq)-1 < changed_position :
            #print("Observed sequence is shorter in length than mutant position! so discarding it.")
            continue
        if seq[changed_position] == changed_to_residue.strip():
            mut_dic[mutant] = changed_position
    #For Multiple res mutant if found 
    for mutant in mul_sub_mutants:
        #Example: KVG444-6TST
        stripped = mutant.split("-")
        first_term = stripped[0] #KVG444
        last_term = stripped[1]  #6TST
        original_residues = re.findall(r"[A-Z]+", first_term)[0] #KVG
        position = int(re.findall(r"\d+", first_term)[0]) - 1 #444 -->443 
        changed_residue = re.findall(r"[A-Z]+", last_term)[0] #TST

        if len(seq)-1 < position + len(changed_residue):
            continue
        if seq[position : position+ len(changed_residue)] ==changed_residue:
            mut_dic[mutant] = changed_position

    return mut_dic

def generate_greany_sig_seqs_from_gsaid():
    greany_seqs_dict = get_greany_sequences_frm_gsaid()
    file_name = GREANY_SEQ_PATH+ os.path.sep + "greany_gsaid.csv"
    tabular_data = []
    for mutant in greany_seqs_dict:
        seqs_per_mutant = greany_seqs_dict[mutant]
        for sequence in seqs_per_mutant:
            row = [mutant, sequence]
            tabular_data.append(row)
    
    dataFrame = pd.DataFrame(tabular_data)
    dataFrame.to_csv(file_name, index=False, header=None)
    print("Greany Sequences Generated Successfully !!!")

def is_greany_significant_seq(seq):
    muts = readAdditionalMutants(GREANY_SIG_MUT_FILE)
    for mutant in muts:
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        if len(seq)-1 < changed_position :
            #print("Observed sequence is shorter in length than mutant position! so discarding it.")
            continue

        if seq[changed_position] == changed_to_residue.strip():
            return (True, mutant)
    return (False, -1)

def get_greany_sequences_frm_gsaid():
    #Stores key = seq, value = Position
    greany_seqs = {   }
    seqs  = SEG.read_sequences(GSAID_SIG_SPIKE_PATH)
    SEQ_LIMIT = 5
    for seq in seqs:
        is_sig, mutant = is_greany_significant_seq(seq)
        if(is_sig == True ):
            if mutant in greany_seqs:
                #No of sequences per mutatant | Add upto 5 sequences per mutant only 
                seqences_per_mutants = greany_seqs[mutant]
                no_of_seqs = len(seqences_per_mutants)
                if no_of_seqs < SEQ_LIMIT:
                    seqences_per_mutants.append(seq)
            else:
                greany_seqs[mutant] = [seq]
       
    print("Total Greany Mutant Keys founds are: ",len(greany_seqs))
    return greany_seqs

def get_windowed_seqs_for_embedded_training():
    windowed_seqs = set()
    step_size = 20
    for record in  SeqIO.parse('data/gsAid-original/spikeprot0327.fasta', 'fasta'):
        #if len(record.seq) < 1000:
            #continue
        if str(record.seq).count('X') > 0:
            continue
        seq = record.seq
        seq =  seq.strip("*")
        end_index = len(seq)
        for i in range(0, end_index, step_size):
            window = str(seq[i : i+step_size])
            if len(window) == step_size:
                windowed_seqs.add(window)
            
    
    print("Total Windows Generated are: ", len(windowed_seqs))
    return list(windowed_seqs)
        
def save_windowed_seqs_for_embedded_training():
    windowed_seqs  = get_windowed_seqs_for_embedded_training()
    df = pd.DataFrame({'window_seqs': (windowed_seqs) })
    df.to_csv(f'data/gen/windowed_embed_train_seqs_{len(windowed_seqs)}.csv', index=False)

def construct_seq_segments(segment_length=20, source_path='', MAX_SEQ=10):
    from random import sample
    MAX_READ_SEQ_FROM_SOURCE= 1000000 #10 lakhs
    seqs_from_sources  = SEG.read_sequences_with_limit(source_path, MAX_READ_SEQ_FROM_SOURCE)
    seqs = sample(seqs_from_sources, MAX_SEQ)
    windowed_seqs = []
    for seq in seqs:
        clean_seq = cleanse_gsaid_seq(seq) #remove trailing * present in gisaid seq
        arr = []
        step_size = segment_length
        end_index = len(seq)
        for i in range(0, end_index, step_size):
            window = clean_seq[i : i+step_size]
            #Ignore seq with length less than 20
            if len(window) < segment_length:
                continue
            windowed_seqs.append(window)
    file_name = f"data/gen/windowed_seqs_{segment_length}_length_windows_{len(windowed_seqs)}.csv"
    CSV_UTIL.write_csv(data_dict={'window_seqs':windowed_seqs }, file_path=file_name)

def construct_windowed_seq_segments(segment_length=20, mutant_file_path='', generate_file_name='', IS_SIG_TYPE=False):
    windowed_seqs = []
    wild_seq  = read_wildSequence()
    wild_seq_len = len(wild_seq)
    #Read mutant file mutant 
    mutants  = readAdditionalMutants(mutant_file_path)
    counter = 0
    for mutant in mutants:
        #Example E848K - Extract E | 848 | K
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        pos =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        #Get data from 20 to 40 | Split wild residue string into the array 
        seq_array = [amino_acid for amino_acid in wild_seq]
        #Invalid mutants has to be discarded 
        if IS_SIG_TYPE == True and  (seq_array[pos]  != original_residue):
            counter = counter +1
            continue
        #assert(seq_array[pos]  == original_residue) #Make sure that original residue must be same in wild seq and original residue

        #replace the mutated residue at postion 
        seq_array[pos] = changed_to_residue

        half_segment_len = int(segment_length/2)
        if pos < half_segment_len:
            split_window = ''.join(seq_array[0 : segment_length])
        elif pos+half_segment_len > len(wild_seq):
            split_window = ''.join(seq_array[wild_seq_len-segment_length : ])
        else: 
            split_window = ''.join(seq_array[pos-half_segment_len : pos+half_segment_len])

        assert(len(split_window) == segment_length)
        windowed_seqs.append(split_window)
    if IS_SIG_TYPE == True:    
        print("Total unmatched original and changed residue in wild seq and mutated mutants in SIG Seqs are : ", counter)
        
    print("Total windowed seqs: ", len(windowed_seqs))
    CSV_UTIL.write_csv(data_dict={'window_seqs':windowed_seqs }, file_path=generate_file_name)
'''
This method reads combined windows file and generates single merged sample window file
'''
def generate_gisaid_non_sig_combined_windowed_seqs_samples(sample_frac=0.25, gen_file_name=''):
    df = pd.read_csv("data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs.csv")
    sample_df = df.sample(frac=sample_frac) #Get only 0.25% of sample  
    save_path = "data/additional_escape_variants/gen/"+gen_file_name
    sample_df.to_csv(save_path, index = False)


def read_window_file(file_path):
    df = pd.read_csv(file_path)
    x = df['window_seqs'].tolist()
    print("Len of file",len(x))
    return x
'''
This reads features from combined window file and returns merged single  list 
The windows are added alteratively starting from wild+mutated ->wild + mutated and so on
All Even Postions = Wild Type | Odd positions = Mutated type
'''
def read_combined_window_file(file_path):
    combined_features = []
    df = pd.read_csv(file_path)
    #Appending features | 
    for row in df.itertuples():
        combined_features.append(row.wild)
        combined_features.append(row.mutated)
    print("Combined Features Length: ", len(combined_features))
    #Ensure that features lenght is always even
    assert(len(combined_features) % 2 == 0)
    return combined_features






'''
Generates windowed sequences of size 20 for both significant and non significant sequences 
The windowed sequences are generated based on single residue substituion information obtained from ESC resource for significant
and Greany non significant mutant resource for non sig window
'''
def generate_esc_windowed_seqs():
    sig_gen_file = ADDITIONAL_MUTANT_GENERATED_PATH + os.path.sep + 'sig_train_mut_window_size_20.csv'  
    construct_windowed_seq_segments(mutant_file_path=SINGLE_RES_TRAIN_MUT_WITHOUT_GREANY, generate_file_name=sig_gen_file, IS_SIG_TYPE = True)
    path_non_sig = 'non_sig_train_mut.csv'
    non_sig_gen_file = ADDITIONAL_MUTANT_GENERATED_PATH + os.path.sep + 'non_sig_train_mut_window_size_20.csv' 
    construct_windowed_seq_segments(mutant_file_path=path_non_sig, generate_file_name=non_sig_gen_file)
'''
Generate windowed sequences with two pair as tuple representating 
(Orignal window, Mutated Window) Seq
'''
def generate_combined_windowed_seqs(gen_file_name, com_windows):
    import pandas as pd
    cols = ['wild', 'mutated']
    df = pd.DataFrame(com_windows, columns=cols)
    gen_path = ADDITIONAL_MUTANT_GENERATED_PATH + os.path.sep + gen_file_name 
    df.to_csv(gen_path, mode='w', index=False)
    print("Windows file generated successfully ! Path: ", gen_path)

def generate_esc_sig_combined_windows(gen_file_name):
    windows = get_esc_sig_combined_windows()
    generate_combined_windowed_seqs(gen_file_name, windows)

def get_esc_sig_combined_windows():
    #Read mutant file mutant 
    mutants  = readAdditionalMutants('single_res_sig_train_mut_without_greany.csv')
    return get_single_residue_combined_windows(mutants)

'''
This is a generic method that constructs paired windows for wild type and mutated type for single residue subsitution mutatnts
-Arg: single_res_mutant_list - This is the mutant list containing mutant info: Ex - ['C250L', 'D456E']
returns list [(wild_type_window, mutated_type_window) ]
'''
def get_single_residue_combined_windows(single_res_mutant_list):
    windowed_seqs = []
    wild_seq  = read_wildSequence()
    for mutant in single_res_mutant_list:
        #Example E848K - Extract E | 848 | K
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        pos =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 

        #Get data from 20 to 40 | Split wild residue string into the array 
        seq_array = [amino_acid for amino_acid in wild_seq]
        wild_window = SEG.extract_window(pos=pos, input_seq=wild_seq)
        
        seq_array[pos] = changed_to_residue
        mutated_seq = ''.join(seq_array)
        mutated_window = SEG.extract_window(pos=pos, input_seq=mutated_seq)
        if len(wild_window) != len(mutated_window):
            print("Caution !!! Mutate window and wild window are of diff length: ")
            continue
        windowed_seqs.append( (wild_window, mutated_window))
    print("Constructed windows length: ", len(windowed_seqs))
    return windowed_seqs

def get_greany_sig_combined_windows():
    #Read mutant file mutant 
    mutants  = readAdditionalMutants(GREANY_SIG_MUT_FILE)
    return get_single_residue_combined_windows(mutants)

def get_greany_non_sig_combined_windows():
    #Read mutant file mutant 
    mutants  = readAdditionalMutants(GREANY_NON_SIG_MUT_FILE)
    return get_single_residue_combined_windows(mutants)

def generate_greany_sig_combined_windows(gen_file_name):
    windows = get_greany_sig_combined_windows()
    generate_combined_windowed_seqs(gen_file_name, windows)


def generate_greany_non_sig_combined_windows(gen_file_name):
    windows  = get_greany_non_sig_combined_windows()
    generate_combined_windowed_seqs(gen_file_name, windows)

#Baum combined windows generation
def generate_baum_sig_combined_windows(gen_file_name):
    mutants  = readAdditionalMutants(BAUM_SIG_MUT_FILE)
    windows =  get_single_residue_combined_windows(mutants)
    generate_combined_windowed_seqs(gen_file_name, windows)

def generate_buam_non_sig_combined_windows(gen_file_name):
    mutants  = readAdditionalMutants(BAUM_NON_SIG_MUT_FILE)
    windows =  get_single_residue_combined_windows(mutants)
    generate_combined_windowed_seqs(gen_file_name, windows)


def generate_gisaid_sig_windows():
    segment_length = 20
    sig_path = SEG.getGsaid_significantSeqs_sample_path()
    significant_seqs = SEG.read_sequences(sig_path)
    windows = []
    for seq in significant_seqs:
        muts_dic = get_mutant_info_from_seq(seq)
        #If mut dict is  empty - continue | {'R346S': 345, 'Y453F': 452}
        if not muts_dic:
            continue
        for mutant, position in muts_dic.items():
            window = SEG.extract_window(segment_length=segment_length, pos=position, input_seq = seq)
            if len(window) != segment_length:
                continue
            windows.append(window)
    print("Total GISAID Significant windows Before duplicate removed: ", len(windows))
    duplicate_removed_windows = list (set (windows))
    file_name = f"data/gen/gisaid_sig_windowed_seqs_{segment_length}_dup_removed_length_windows_{len(duplicate_removed_windows)}.csv"
    CSV_UTIL.write_csv(data_dict={'window_seqs':duplicate_removed_windows }, file_path=file_name)
    print(f"GISAID file with path {file_name} generated successfully")

def generate_gisaid_sig_comb_windows(gen_file_name):
    windows = get_gisaid_sig_comb_windows()
    generate_combined_windowed_seqs(gen_file_name, windows)
'''
Removes greany sequences if found in training dataset 
'''
def generate_gisaid_sig_comb_windows_after_greany_removal():
    df_gisaid_sig_bf_cleansing = pd.read_csv("data/additional_escape_variants/gen/gisaid_sig_combined_windowed_seqs-before-cleansing.csv") 
    df_greany_sig = pd.read_csv("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv") 

    greany_sig_set = set()
    for index, row in df_greany_sig.iterrows():
        greany_sig_set.add( (row['wild'], row['mutated']) )

    gisaid_sig_set = set()
    for index, row in df_gisaid_sig_bf_cleansing.iterrows():
        gisaid_sig_set.add( (row['wild'], row['mutated']) )

    sig_set = list (gisaid_sig_set - greany_sig_set)
    df_new  = pd.DataFrame(sig_set, columns=['wild', 'mutated'])
    df_new.to_csv('data/additional_escape_variants/gen/gisaid_sig_combined_windowed_seqs_after_greany_removal.csv', index = False)
#
def generate_gisaid_sig_comb_windows_after_greany_and_baum_removal():
    df_gisaid_sig_bf_cleansing = pd.read_csv("data/additional_escape_variants/gen/gisaid_sig_combined_windowed_seqs-before-cleansing.csv") 
    df_greany_sig = pd.read_csv("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv") 
    df_baum_sig = pd.read_csv("data/additional_escape_variants/gen/baum_sig_combined_windowed_seqs.csv") 

    greany_sig_set = set()
    for index, row in df_greany_sig.iterrows():
        greany_sig_set.add( (row['wild'], row['mutated']) )

    baum_sig_set = set()
    for index, row in df_baum_sig.iterrows():
        baum_sig_set.add( (row['wild'], row['mutated']) )

    gisaid_sig_set = set()
    for index, row in df_gisaid_sig_bf_cleansing.iterrows():
        gisaid_sig_set.add( (row['wild'], row['mutated']) )

    sig_set = list (gisaid_sig_set - greany_sig_set - baum_sig_set)
    df_new  = pd.DataFrame(sig_set, columns=['wild', 'mutated'])


    df_new.to_csv('data/additional_escape_variants/gen/gisaid_sig_combined_windowed_seqs_after_greany_and_baum_removal.csv', index = False)

'''
Removes baum and greany windows 
and generated sig windows using esc and gisaid seqs
'''
def generate_sig_combined_windows():
    baum_frame = pd.read_csv('data/additional_escape_variants/gen/baum_sig_combined_windowed_seqs.csv', header=0)
    baum_sig_set = set()
    for index, row in baum_frame.iterrows():
        baum_sig_set.add( (row['wild'], row['mutated']) )

    #Read esc sig mut dataset 
    esc_frame = pd.read_csv('data/additional_escape_variants/gen/esc_sig_combined_windowed_seqs.csv', header=0)
    esc_sig_muts = esc_frame.iloc[:, 0].to_list()
    print("ESc Sig Muts Before: ", len(esc_sig_muts))
    esc_sig_muts = set()
    for index, row in esc_frame.iterrows():
        esc_sig_muts.add( (row['wild'], row['mutated']) )

    #read gisaid  mut dataset
    gisaid_frame = pd.read_csv('data/additional_escape_variants/gen/gisaid_sig_combined_windowed_seqs_after_greany_and_baum_removal.csv', header=0)
    gisaid_windows = gisaid_frame.iloc[:, 0].to_list()
    print("gisaid_windows Sig Muts: ", len(gisaid_windows))
    gisaid_sig_muts = set()
    for index, row in gisaid_frame.iterrows():
        gisaid_sig_muts.add( (row['wild'], row['mutated']) )

    #combined esc and gisaid dataset (note that esc alos makes sure that no greany data are present)
    comb_sig = gisaid_sig_muts.union(esc_sig_muts)
    print("Combined Sig Union", len(comb_sig))

    sig_set = list (comb_sig - baum_sig_set)
    print("Sig set after removal of baum", len(sig_set))

    df_new  = pd.DataFrame(sig_set, columns=['wild', 'mutated'])
    df_new.to_csv('data/additional_escape_variants/gen/sig_combined_windows.csv', index = False)







'''
Check if any significant window from (GISAID/ Greany / or greany non sig window ) is present there, 
if it is present removes these and returns gisaid non sig windows
'''
def get_cleansed_gisaid_non_sig_combined_windows(gisAid_windows):
    df_greany_sig = pd.read_csv("data/additional_escape_variants/gen/greany_sig_combined_windowed_seqs.csv") 
    df_gisaid_sig = pd.read_csv("data/additional_escape_variants/gen/gisaid_sig_combined_windowed_seqs.csv") 
    df_esc_sig = pd.read_csv("data/additional_escape_variants/gen/esc_sig_combined_windowed_seqs.csv") 
    df_greany_non_sig = pd.read_csv("data/additional_escape_variants/gen/greany_non_sig_combined_windowed_seqs.csv")
    
    data_to_be_removed = pd.concat([df_esc_sig, df_gisaid_sig, df_greany_sig, df_greany_non_sig] , axis = 0)
    b_set = convert_combined_win_dataframe_to_set(data_to_be_removed)

    a_set = set(gisAid_windows)
    
    diff_set = list (a_set - b_set)
    print("Lenght of diff set is : ", len(diff_set))
    return diff_set

def convert_combined_win_dataframe_to_set(df):
    window_set = set()
    for index, row in df.iterrows():
        window_set.add( (row['wild'], row['mutated']) )
    return window_set





def get_gisaid_sig_comb_windows():
    segment_length = 20
    wild_seq  = read_wildSequence()
    sig_path = SEG.getGsaid_significantSeqs_sample_path()
    significant_seqs = SEG.read_sequences(sig_path)
    windows = []
    for seq in significant_seqs:
        muts_dic = get_mutant_info_from_seq(seq)
        #If mut dict is  empty - continue | {'R346S': 345, 'Y453F': 452}
        if not muts_dic:
            continue
        for mutant, position in muts_dic.items():
            mut_window = SEG.extract_window(segment_length=segment_length, pos=position, input_seq = seq)
            if len(mut_window) != segment_length:
                continue
            wild_window = SEG.extract_window(segment_length=segment_length, pos=position, input_seq = wild_seq)
            if len(mut_window) == len(wild_window) == segment_length:
                windows.append( (wild_window, mut_window))
    duplicate_removed_windows = list (set (windows))
    print(f"Total GISAID Significant windows Before duplicate removed: { len(windows)} After duplicate removed: { len(duplicate_removed_windows)}")
    return duplicate_removed_windows


def generate_gisaid_non_sig_windows():
    segment_length = 20
    non_sig_path = SEG.getGsaid_non_significant_seq_path()
    non_significant_seqs = SEG.read_sequences(non_sig_path)
    windows = []
    for seq in non_significant_seqs:
        rand_positions = SEG.get_random_positions(max_len = len(seq), total_pos = 66)
        for pos in rand_positions:
            window = SEG.extract_window(segment_length=segment_length, pos=pos, input_seq = seq)
            if len(window) != segment_length:
                continue
            windows.append(window)

    print("Total GSAID Non Sig windows : ", len(windows))
    file_name = f"data/gen/gisaid_non_sig_windowed_seqs_{segment_length}_length_windows_{len(windows)}.csv"
    CSV_UTIL.write_csv(data_dict={'window_seqs':windows }, file_path=file_name)
    print(f"GISAID file with path {file_name} generated successfully")

def get_gisaid_non_sig_combined_windows(no_split_positions):
    segment_length = 20
    wild_seq  = read_wildSequence()

    non_sig_path = SEG.getGsaid_non_significant_seq_path()
    non_significant_seqs = SEG.read_sequences(non_sig_path)
    windows = []
    sig_positions = get_significant_positions()
    for seq in non_significant_seqs:
        rand_positions = SEG.get_random_positions(max_len = len(seq), total_pos = no_split_positions)
        #Get random positions other than signficant positions
        rand_positions = list ( set(rand_positions) - set(sig_positions) )
        for pos in rand_positions:
            mut_window = SEG.extract_window(segment_length=segment_length, pos=pos, input_seq = seq)
            
            wild_window = SEG.extract_window(segment_length=segment_length, pos=pos, input_seq = wild_seq)
            if len(mut_window) == len(wild_window) == segment_length:
                    windows.append( (wild_window, mut_window))

            
    duplicate_removed_windows = list (set (windows))
    print("Total GSAID Non Sig Combined windows : ", len(duplicate_removed_windows))
    return duplicate_removed_windows

def generate_gisaid_non_sig_combined_windows(split_positions = 100):
    windows = get_gisaid_non_sig_combined_windows(split_positions)
    print("Length of Non Sig window before cleansing: ", len(windows))

    gen_file_name = f"gisaid_non_sig_combined_windowed_seqs_{len(windows)}.csv"

    cleansed_window = get_cleansed_gisaid_non_sig_combined_windows(windows)
    generate_combined_windowed_seqs(gen_file_name, cleansed_window)



'''
Given input list of ['A245G', 'C344G'] 
returns 244, 343 | (As computer index starts from 0 , the number is returned -1)
'''
def get_single_residue_positions(muts):
    positions = []
    for mutant in muts:
        original_residue = mutant[0]
        changed_to_residue= mutant[-1]
        changed_position =   int(mutant[1:-1])  - 1 #Index starts from 0, so 1 is subtracted 
        positions.append(changed_position)
    return positions


def generate_all_signficant_positions():
    sig_positions = [444, 445, 446, 455, 456]
    greany_positions = get_single_residue_positions(readAdditionalMutants(GREANY_SIG_MUT_FILE))
    esc_positions = get_single_residue_positions(readAdditionalMutants(MutationTypes.SINGLE_RES_SUB.name+".csv"))

    combined_pos = sig_positions + greany_positions + esc_positions
    positions = list (set(combined_pos))
    print(f"Sig Position with  duplicates {len(combined_pos)}********* After: {len(positions)}")
    file_path = ADDITIONAL_MUTANT_GENERATED_PATH + os.path.sep+'signficant_positions.csv'
    CSV_UTIL.write_csv({'pos': positions }, file_path)
def get_significant_positions():
    file = ADDITIONAL_MUTANT_GENERATED_PATH + os.path.sep+'signficant_positions.csv'
    df  = pd.read_csv(file)
    positions = df['pos'].to_list()
    print("Total Signficant Postions: ", len(positions))
    return positions

def create_sharded_non_sig_gisaid_dataset(total_segments=18):
    hf1 = h5py.File('data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_652734.h5', 'r')
    non_sig_hdf = hf1.get('non_sig_seqs')
    non_sig_arry = np.array(non_sig_hdf)  #632734 : Total seqs 
    hf1.close()
    
    #Create new sharded file
    hf =  h5py.File('data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_sharded.h5', 'w')
    non_sig_group = hf.create_group('NON_SIG')
    for i in range(total_segments):
        data_set = f"dataset_{i}"
        data_segment = get_data_segment(total_segments, i, non_sig_arry)
        non_sig_group.create_dataset(data_set, data = data_segment)
    
    hf.close()
    print("Sharded Non GISAID dataset created successfully.")

def get_data_segment(total_segments, segment_no , data):
    size_of_split = int (len(data)/total_segments)
    if size_of_split % 2 != 0:
        size_of_split = size_of_split + 1
    
    if segment_no + 1 == total_segments:
        return data[segment_no*size_of_split :  ]
    else: 
        return data [segment_no * size_of_split : (segment_no+1) * size_of_split ]
'''
Gets the split window of fixed size for specific lenght 
returns tuple of base and offset 
'''
def get_data_segment_indexes(total_segments = 0,  total_data_length = 0):
    if total_segments == 0 or total_data_length == 0 :
        print("Error ! specify total segments and total data length ")
    segments = []
    size_of_split = int (total_data_length/total_segments)
    for i in range(total_segments):
        base = i*size_of_split
        offset = (i+1)*size_of_split
        if i == total_segments -1:
            segments.append( (base, total_data_length ) )
        else:
            segments.append( (base, offset) )
    
    return segments





def read_sharded_gisaid_non_sig_seqs():
    hf =  h5py.File('data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_sharded.h5', 'r')
    non_sig_data = hf['NON_SIG']
    for dataset_key in non_sig_data.keys():
        data = np.array(non_sig_data[dataset_key])
        print(type(data), "Shape: ",data.shape)
        break
def get_gisaid_nonsig_shared_dataset_instance():
    global hf
    if hf is None:
        hf =  h5py.File(GISAID_NON_SIG_COMBINED_WINDOWED_SEQS_SHARDED_PATH, 'r')
    return hf

def get_gisaid_nonsig_sharded_dataset_keys(hf_instance):
      return  list(hf_instance['NON_SIG'].keys())
    
def transform_reduced_non_sig_embed_features():
    hf = get_gisaid_nonsig_shared_dataset_instance()
    keys  = get_gisaid_nonsig_sharded_dataset_keys(hf)
    count = 0
    INITIALIZED= False
    for key in keys:
        dataset = get_sharded_data_by_key(hf, key)
        reduced_fzs  = reduce_features(dataset) 
        if  INITIALIZED == False: 
            npArray = reduced_fzs
            INITIALIZED = True
        else: 
            npArray = np.concatenate( (npArray, reduced_fzs) , axis=0 )
        #Freeing up memory
        del dataset 
        del reduced_fzs 

    print ("Shape of concatenated reduced gisaid non sig features  :  ", npArray.shape)
    hf.close()
    return npArray
def get_combined_features(save_path):
    data = np.load(save_path)
    features = data['arr_0']
    print(f"Retrieved features shape: {features.shape}")
    return features
def transform_reduced_greany_sig_embed_features():
    #Sig features from esc resource and gisaid
    sig_features = get_combined_features(FEATURE_GREANY_SIG_COMBINED_PATH)
    non_sig_features  = get_combined_features(FEATURE_GREANY_NON_SIG_COMBINED_PATH)

    sig_reduced_data  =  reduce_features(sig_features)
    non_sig_reduced_data  =  reduce_features(non_sig_features)
    print(f"Reduced greany sig  shape{sig_reduced_data.shape}  nonsig dataset shape : {non_sig_reduced_data.shape} ", )

    return sig_reduced_data, non_sig_reduced_data

def transform_reduced_sig_embed_features():
    #Sig features from esc resource and gisaid
    sig_features_esc = get_combined_features(FEATURE_ESC_SIG_COMBINED_PATH)
    sig_gisiad_features  = get_combined_features(FEATURE_GISAID_SIG_COMBINED_PATH)

    sig_features  =  np.concatenate( (sig_features_esc, sig_gisiad_features) , axis= 0 )
    print("Sig aggretate features shape: ", sig_features.shape)

    reduced_data  =  reduce_features(sig_features)
    print("Reduced dataset shape for sig embed features shape: ", reduced_data.shape)
    del sig_features_esc, sig_gisiad_features, sig_features #Freeing space
    return reduced_data


def save_reduced_sharded_non_sig_embed_features():
    red_features = transform_reduced_non_sig_embed_features()
    np.savez_compressed(REDUCED_NONSIG_FEATURE_PATH, DATA = red_features)
    print("Features Saved successfully, saved features shape: ", red_features.shape)
    del red_features

def save_reduced_non_shared_sig_embed_features(input_file_path ,output_file_path):
    hf =  h5py.File(input_file_path, 'r')
    dataset = np.array(hf['non_sig_seqs'])
    reduced_fzs  = reduce_features(dataset) 
    print ("Shape of  reduced gisaid non sig features  :  ", reduced_fzs.shape)
    
    np.savez_compressed(output_file_path, DATA = reduced_fzs)
    print("Features Saved successfully, saved features shape: ", reduced_fzs.shape)
    hf.close()
    del reduced_fzs

def save_reduced_sig_embed_features():
    red_features = transform_reduced_sig_embed_features()
    np.savez_compressed(REDUCED_SIG_FEATURE_PATH, DATA = red_features)
    print("Features Saved successfully, saved features shape: ", red_features.shape)
    del red_features

def save_reduced_greany_embed_features():
    sig_reduced_data, non_sig_reduced_data = transform_reduced_greany_sig_embed_features()
    np.savez_compressed(REDUCED_GREANY_FEATURE_PATH, SIG_DATA = sig_reduced_data, NON_SIG_DATA=non_sig_reduced_data)
    print("Features Saved successfully to Path: ", REDUCED_GREANY_FEATURE_PATH)
    del sig_reduced_data, non_sig_reduced_data

def get_reduced_greany_embed_features():
    data_dic = np.load(REDUCED_GREANY_FEATURE_PATH)
    sig_data = data_dic['SIG_DATA']
    non_sig_data = data_dic['NON_SIG_DATA']
    print(f"Retrived sig features shape: {sig_data.shape} Non sig shape: {non_sig_data.shape}  ")
    del data_dic
    return sig_data, non_sig_data

def get_reduced_baum_embed_features():
    sig_data = np.load(REDUCED_BAUM_SIG_FEATURE_PATH)['arr_0']
    non_sig_data = np.load(REDUCED_BAUM_NON_SIG_FEATURE_PATH)['arr_0']

    sig_data = np.reshape(sig_data , (-1,  4096) )
    non_sig_data = np.reshape(non_sig_data , (-1,  4096) )

    print(f"Retrived sig features shape: {sig_data.shape} Non sig shape: {non_sig_data.shape}  ")
    return sig_data, non_sig_data    

def get_reduced_sig_embed_features():
    data_dic = np.load(REDUCED_SIG_FEATURE_PATH)
    data = data_dic['DATA']
    print("Retrived sig features shape: ", data.shape)
    del data_dic
    return data

def get_reduced_sig_embed_features_new():
    data = np.load(REDUCED_SIG_FEATURE_NEW_PATH)['arr_0']
    print("Reduced Sig feature new shape Retrived sig features shape: ", data.shape)
    return data

def get_reduced_non_sig_embed_features(reduced_non_sig_path):
    
    data_dic = np.load(reduced_non_sig_path)
    data = data_dic['DATA']
    print("Retrived Non sig features shape: ", data.shape)
    del data_dic
    return data

'''
Reduce high dimensional features 45056 *2 dim to 4096 dimension
'''
def reduce_features(dataset):
    x_reshape = int (dataset.shape[0]/2)
    y_middle_shape = 22
    y_reshape = int (2*(dataset.shape[1])/y_middle_shape)
        # (9066, 45056) reshaped to 9066/2, 45056*2 =>(4533, 90112)  ==>4533, 22, 4096
    print(f"and reshaping to : { (x_reshape, y_reshape)} ")
    dataset = np.reshape (dataset, (x_reshape, y_middle_shape, y_reshape) )
    print("Transformed Shape: ", dataset.shape)

    reduced_dataset = np.average(dataset, axis=1)
    print("Reduced feature Shape: ", reduced_dataset.shape)
    
    return reduced_dataset

def split_train_test(df):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, shuffle=True )
    print("Lenght of train data :",train_data.shape )
    print("Shape of test data :",test_data.shape)
    return train_data, test_data  
def save_train_test_datasets():
    non_sig_features  =  pd.read_csv("data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_181887.csv")
    sig_features  = pd.read_csv("data/additional_escape_variants/gen/sig_combined_windows.csv")
    sig_train, sig_test = split_train_test(sig_features)


    save_train_test_windows('sig_combined_windows_train.csv', sig_train)
    save_train_test_windows('sig_combined_windows_test.csv', sig_test)

    non_sig_train, non_sig_test = split_train_test(non_sig_features)
    save_train_test_windows('non_sig_combined_windows_train.csv', non_sig_train)
    save_train_test_windows('non_sig_combined_windows_test.csv', non_sig_test)
    print("All train test dataset saved successfully.")

def save_train_test_windows(file_path, df):
    file_path =  "data/additional_escape_variants/gen/" + str(file_path)
    df.to_csv(file_path, index=False)      




def get_sharded_data_by_key(hf, key):
    non_sig_data = np.array(hf['NON_SIG'][key])
    print(f"Obtained data for Key: {key} , Dataset shape: {non_sig_data.shape} ")
    return non_sig_data


if __name__ == "__main__":
    #wildSeq = read_wildSequence()
    #generate_single_residue_mutation(wildSeq)
    #generate_multiple_residue_subsitution(wildSeq)
    #generate_single_residue_del_seqs(wildSeq)
    
    #generate_baum_sequences("baum.csv")
    #generate_greany_sig_seqs_from_gsaid()
    
    #generate_gsaid_sequences('GISAID_SIG.fa', 'GISAID_NON_SIG.fa')   

    #construct_seq_segments
    #construct_seq_segments(segment_length=20, source_path = SEG.getGsaid_raw_seq_path(), MAX_SEQ=2)

    #generate_esc_windowed_seqs()
    #generate_esc_sig_combined_windows('esc_sig_combined_windowed_seqs.csv')
    #generate_gisaid_sig_comb_windows('gisaid_sig_combined_windowed_seqs-before-cleansing.csv') #
    
    #generate_gisaid_sig_comb_windows_after_greany_removal()
    #generate_gisaid_sig_comb_windows_after_greany_and_baum_removal()
    #generate_sig_combined_windows()

    #path_greany_sig = 'greany_significant_mutants.csv'
    #greany_sig_window_gen_file = ADDITIONAL_MUTANT_GENERATED_PATH + os.path.sep + 'greany_sig_mut_window_size_20.csv' 
    #construct_windowed_seq_segments(mutant_file_path=path_greany_sig, generate_file_name=greany_sig_window_gen_file, IS_SIG_TYPE=True)
    #generate_gisaid_sig_windows()
    #generate_gisaid_non_sig_windows()

    #generate_all_signficant_positions()
    #get_significant_positions()

    #generate_gisaid_non_sig_combined_windows(split_positions=40)

    #generate_greany_sig_combined_windows('greany_sig_combined_windowed_seqs.csv')
    #generate_greany_non_sig_combined_windows('greany_non_sig_combined_windowed_seqs.csv')

    #generate_baum_sig_combined_windows('baum_sig_combined_windowed_seqs.csv')
    #generate_buam_non_sig_combined_windows('baum_non_sig_combined_windowed_seqs.csv')
    #generate_gisaid_non_sig_combined_windowed_seqs_samples(gen_file_name='gisaid_non_sig_combined_windowed_SAMP_200000.csv')


    #create_sharded_non_sig_gisaid_dataset(total_segments=TOTAL_NO_OF_SHARDS)
    #read_shared_gisaid_non_sig_seq()
    #hf, keys  = get_gisaid_nonsig_sharded_dataset_keys()
    #dataset_01  = np.array(hf['NON_SIG']['dataset_0'])

    #print(get_data_segment_indexes(5, 25))

    #save_reduced_sharded_non_sig_embed_features()
    #gisaid_non_sig_combined_windowed_seqs_181887.h5 | 
    #input_path = "data/additional_escape_variants/gen/gisaid_non_sig_combined_windowed_seqs_153140.h5"
    #reduced_non_sig_features_181887.npz
    #output_gen_path = "data/additional_escape_variants/gen/reduced_non_sig_features_153140.npz"
    #save_reduced_non_shared_sig_embed_features(input_path, output_gen_path)
    
    #save_reduced_sig_embed_features()
    #save_reduced_greany_embed_features()

    #get_reduced_baum_embed_features()

    #get_reduced_sig_embed_features_new()
    #save_windowed_seqs_for_embedded_training()

    windows = get_gisaid_non_sig_combined_windows(60)
    cleansed_window = get_cleansed_gisaid_non_sig_combined_windows(windows)




    pass



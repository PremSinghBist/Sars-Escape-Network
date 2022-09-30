from escape  import loadAdditionalEscapes
import SignficantEscapeGenerator as SEG
import AA_change_patterns as patterns
import os
from MutationTypes import MutationTypes
from escape import BASE_DATA_PATH as BASE_PATH
import csv
import escape as ESC
import pandas as pd
import utils 

def get_mutant_dic():
    mutantList = loadAdditionalEscapes()
    print("Mutation Length List:", len(mutantList))
    for mutant in mutantList:
        patterns.add_mutation(mutant)
    mutantDictionary = patterns.getMutationDictionary()
    return mutantDictionary

def print_mutant_summary(mutantDictionary):
    for mutant_type, data in mutantDictionary.items(): 
       print("Mutant type: {} , Total mutants: {}".format(mutant_type.name, len(data)))

def getUniqueItemsFromList(list_items):
    unique_list = []
    [unique_list.append(n) for n in list_items if n not in unique_list]
    return unique_list

def generate_mutant_dataset(getUniqueItemsFromList, mutantDictionary):
    for mutantType, mutantList in mutantDictionary.items():
        mutant_data_path =  BASE_PATH + os.path.sep + "additional_escape_variants"+os.path.sep+"gen"+os.path.sep+mutantType.name+".csv"
        print("*Mutant data Path**:",mutant_data_path)
        with open(mutant_data_path, 'w', encoding='UTF-8', newline="") as f:
            uniqueMutantsList = getUniqueItemsFromList(mutantList) 
            writer = csv.writer(f)
            #Constructed 2D list
            for mutant in uniqueMutantsList:
                writer.writerow([mutant])
'''
This generates Significant and Non signficant mutant  files 
@Significant Mutants CSV: These are single residue substituion mutants after removing greany mutants
@Non Sig Mutants CSV: These are the greany non sig mutants but removed few signficant appeared mutants 
'''
def generate_sig_non_sig_train_mutants():
    greany_sig_muts = ESC.readAdditionalMutants(ESC.GREANY_SIG_MUT_FILE)
    greany_non_sig_muts = ESC.readAdditionalMutants(ESC.GREANY_NON_SIG_MUT_FILE)

    single_res_mutants = ESC.readAdditionalMutants(ESC.MutationTypes.SINGLE_RES_SUB.name+".csv")

    #Single res Training mutant are made sure to remove greany muts 
    print(" SIG single_res_train_mut ")
    single_res_train_mut = utils.diff_operation(single_res_mutants, greany_sig_muts)

    data_dict = {'': single_res_train_mut} 
    path = ESC.ADDITIONAL_MUTANT_GENERATED_PATH+os.path.sep+'single_res_sig_train_mut.csv'
    write_to_csv(data_dict, path)
    
    print(" Non SIG single_res_train_mut ")
    non_sig_train_mut = utils.diff_operation(greany_non_sig_muts, single_res_mutants)
    data_dict = {'': non_sig_train_mut} 
    path = ESC.ADDITIONAL_MUTANT_GENERATED_PATH+os.path.sep+'non_sig_train_mut.csv'
    write_to_csv(data_dict, path)
'''
Finds signficant mutants exists in non significnat greany dataset and save the list
'''
def save_greany_sig_mutants_found_in_non_sig():
    greany_non_sig_muts = ESC.readAdditionalMutants(ESC.GREANY_NON_SIG_MUT_FILE)
    single_res_mutants = ESC.readAdditionalMutants(ESC.MutationTypes.SINGLE_RES_SUB.name+".csv")

    #These are the common mutants ie. common significant mutatns found in non sig seq and sig mutant : ie operation is intersection
    sig_muts_found_in_non_sig = list (set(greany_non_sig_muts) &  set(single_res_mutants))  #& is a set intersection operator
    print("Total sig mutants found in non sig mutants are : ",len(sig_muts_found_in_non_sig))
    data_dict = {'': sig_muts_found_in_non_sig} 
    path = ESC.ADDITIONAL_MUTANT_GENERATED_PATH+os.path.sep+'greany_sig_muts_found_in_non_sig_muts.csv'
    write_to_csv(data_dict, path)

def write_to_csv(data_dict, path):
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(path, header=False,  index=False)


if __name__ == "__main__":
    #mutantDictionary = get_mutant_dic()
    #print_mutant_summary(mutantDictionary)
    #generate_mutant_dataset(getUniqueItemsFromList, mutantDictionary)
    save_greany_sig_mutants_found_in_non_sig()
    generate_sig_non_sig_train_mutants()
    save_greany_sig_mutants_found_in_non_sig()
    pass
    




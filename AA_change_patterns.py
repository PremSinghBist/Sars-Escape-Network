import re
from MutationTypes import MutationTypes 
MUTS = {
       MutationTypes.SINGLE_RES_SUB : [],
       MutationTypes.MULTIPLE_RES_SUB: [],

       MutationTypes.DEL_SINGLE_RES: [],
       MutationTypes.DEL_RES_1_OR_2: [],
       MutationTypes.DEL_IN_RANGE: [],
       #Frequent Delete
       MutationTypes.DEL_SINGLE_RES_FREQ: [],
       MutationTypes.DEL_IN_RANGE_FREQ: [],
       #Insert Inbetween
       MutationTypes.INSERT_IN_BETWEEN: []
}
def getMutationDictionary():
    return MUTS


'''
substitution_pattern3 = "A55L"

'''
def single_residue_subsitution(pattern):
#pattern integer-integerdel - del is found at last  .? - o or 1 chara
#Begin with(^D : No digit max len=1) ->one or more digits(\d+)-> End with no digit max len=1
#. =>Represents any char ie it represents number too. , so dot can not be used
    if re.match("^\D{1}\d+\D{1}$", pattern):
        #print("Patterns like P314L  found")
        return MutationTypes.SINGLE_RES_SUB
    return MutationTypes.NOT_RECOGNIZED

'''
Example: AA55-6TS
'''
def multiple_residue_subsitution(multiple_substitution_pattern3):
    #
    if re.match("^[A-Z]+[0-9]+[-][0-9]+[A-Z]+$", multiple_substitution_pattern3):
        #print("Patterns   found : ", multiple_substitution_pattern3)
        return MutationTypes.MULTIPLE_RES_SUB
    return MutationTypes.NOT_RECOGNIZED

'''
  Pattern -> Î”141-144 =>Δ | frequent delete
'''
def frequent_range_delete(pattern1):
    if re.match("^Δ\d+[-]\d+$", pattern1):
        #print("Patterns   found : ", pattern1)
        return MutationTypes.DEL_IN_RANGE_FREQ
    return MutationTypes.NOT_RECOGNIZED

'''
Pattern -> Î”146  =>Δ | Frequent delete
'''
def freqeuent_single_residue_delete(pattern2):
    if re.match("^Δ\d+$", pattern2):
        #print("Patterns   found : ", pattern2)
        return MutationTypes.DEL_SINGLE_RES_FREQ
    return MutationTypes.NOT_RECOGNIZED

def delete_pattern_extractor(pattern):
    #pattern integer-integerdel - del is found at last | 3675-3677del | f141del
    if delFound_at_end(pattern):
        #print("Del Pattern match at the end")
        #3675-3677del 
        if is_hypen_found(pattern):
            #print("3675-3677del Pattern match")
            return MutationTypes.DEL_IN_RANGE
    #Any alphabet . then any digit | f141del
        elif is_alphabet_follows_digits(pattern):
            #print("Pattern f141del matched ")
            return MutationTypes.DEL_SINGLE_RES
    #if  "del" is found at beginning | del241/243
    elif is_del_in_beginning(pattern):
        if is_nums_slash_nums(pattern):
            #print("del241/243 pattern match")
            return MutationTypes.DEL_RES_1_OR_2
        else:
            return MutationTypes.NOT_RECOGNIZED
    else:
        return MutationTypes.NOT_RECOGNIZED

def is_nums_slash_nums(pattern):
    if re.findall("[0-9][/][0-9]", pattern):
        return True
    else: 
        return False

def is_del_in_beginning(pattern):
    if re.findall("\Adel", pattern):
        return True
    else:
        return False

def is_alphabet_follows_digits(pattern):
    if re.findall("[A-Z][0-9]+", pattern):
        return True
    return False

def is_hypen_found(pattern):
    if re.findall("[-]", pattern):
        return True
    return False

def delFound_at_end(pattern):
    if re.findall("del\Z", pattern):
        return True
    return False


def insert_in_between(pattern):
    if re.match("^\d+[a-z][A-Z]+\d+[a-z]$", pattern):
        #print(" found", pattern)
        return MutationTypes.INSERT_IN_BETWEEN
    return MutationTypes.NOT_RECOGNIZED
'''
This method recognize the spcific type of mutation such as insert/delete etc 
and adds to the mutation dictionary
-Input : Specific Mutant such as : E484K 
 -Output : Addition of one more single subsitution mutation to dictionary
  Ex- {'MutationTypes.SINGLE_RES_SUB' : [E46k, D586I, C345O] }
'''

def add_mutation(mutant):
    #Subsitution
    if MutationTypes.SINGLE_RES_SUB is single_residue_subsitution(mutant):
        MUTS.get(MutationTypes.SINGLE_RES_SUB).append(mutant)
        return 
    elif MutationTypes.MULTIPLE_RES_SUB is multiple_residue_subsitution(mutant):
        MUTS.get(MutationTypes.MULTIPLE_RES_SUB).append(mutant)
        return
    #Frequent Deletion
    elif MutationTypes.DEL_SINGLE_RES_FREQ is freqeuent_single_residue_delete(mutant):
        MUTS.get(MutationTypes.DEL_SINGLE_RES_FREQ).append(mutant)
        return
    elif MutationTypes.DEL_IN_RANGE_FREQ is frequent_range_delete(mutant):
        MUTS.get(MutationTypes.DEL_IN_RANGE_FREQ).append(mutant)
        return
    #Deletion
    elif MutationTypes.DEL_SINGLE_RES is delete_pattern_extractor(mutant):
        MUTS.get(MutationTypes.DEL_SINGLE_RES).append(mutant)
        return
    elif MutationTypes.DEL_RES_1_OR_2 is delete_pattern_extractor(mutant):
        MUTS.get(MutationTypes.DEL_RES_1_OR_2).append(mutant)
        return
    elif MutationTypes.DEL_IN_RANGE is delete_pattern_extractor(mutant):
        MUTS.get(MutationTypes.DEL_IN_RANGE).append(mutant)
        return

    #Insertion
    elif MutationTypes.INSERT_IN_BETWEEN is insert_in_between(mutant):
        MUTS.get(MutationTypes.INSERT_IN_BETWEEN).append(mutant)
        return


if __name__ == "__main__":
    #loadAdditionalEscapes()
    #sequence2  = add_mutation("E525K")
    print("test")
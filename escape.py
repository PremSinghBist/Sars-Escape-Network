import os
from utils import Seq, SeqIO
import csv
from MutationTypes import MutationTypes
import pandas as pd
BASE_DATA_PATH='/home/perm/cov/data'
GENE_TYPE_ORF = "ORF1a"
GENE_TYPE_SPIKE = "S"
ADDITIONAL_ESCAPE_SOURCE = BASE_DATA_PATH+'/additional_escape_variants/additional_sig_escapes.tsv'
ADDITIONAL_MUTANT_GENERATED_PATH = BASE_DATA_PATH +'/additional_escape_variants/gen'
ESCAPE_ANALYSIS_PATH = BASE_DATA_PATH + '/analysis'
BAUM_SEQ_PATH = BASE_DATA_PATH+ os.path.sep + "gen" + os.path.sep+"baum"
GREANY_SEQ_PATH = BASE_DATA_PATH+ os.path.sep + "gen" + os.path.sep+"greany"


BAUM_CSV_FILE_NAME = "baum.csv"
BAUM_SIG_MUT_FILE = "baum_significant_mutants.csv"
BAUM_NON_SIG_MUT_FILE = "baum_non_sig_muts_cleansed.csv" 

GREANY_GISAID_CSV_FILE_NAME = "greany_gsaid.csv"

GREANY_SIG_MUT_FILE = "greany_significant_mutants.csv"
GREANY_NON_SIG_MUT_FILE = "greany_not_significant_mutants.csv"




SEP = os.path.sep

#SIGNIFICANT Mutation Postions from escape resource data source
MUL_RES_MUT_POSITIONS = [444, 445, 446, 455, 456]
DEL_RES_MUT_POSITIONS= [140, 144, 145]
INS_RES_MUT_POSITIONS = [248]

def load_baum2020():
    seq = read_wildSequence()
    #These are the verified escape mutants - example E484K is an escape mutation which helps virus to escape standing army
    muts = set([
        'K417E', 'K444Q', 'V445A', 'N450D', 'Y453F', 'L455F',
        'E484K', 'G485D', 'F486V', 'F490L', 'F490S', 'Q493K',
        'H655Y', 'R682Q', 'R685S', 'V687G', 'G769E', 'Q779K',
        'V1128A'
    ])

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V',
    ]

    seqs_escape = {}
    for idx in range(len(seq)):
        for aa in AAs:
            if aa == seq[idx]:
                continue
            mut_seq = seq[:idx] + aa + seq[idx+1:]
            mut_str = '{}{}{}'.format(seq[idx], idx + 1, aa)
            if mut_seq not in seqs_escape:
                seqs_escape[mut_seq] = []
            seqs_escape[mut_seq].append({
                'mutation': mut_str,
                'significant': mut_str in muts,
            })

    return seq, seqs_escape

def read_wildSequence():
    seq = SeqIO.read(BASE_DATA_PATH+'/cov2_spike_wt.fasta', 'fasta').seq
    return seq

def load_greaney2020(survival_cutoff=0.3,
                     binding_cutoff=-2.35, expr_cutoff=-1.5):
    seq = SeqIO.read(BASE_DATA_PATH+'/cov2_spike_wt.fasta', 'fasta').seq

    sig_sites = set()
    with open(BASE_DATA_PATH+'/significant_escape_sites.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            sig_sites.add(int(fields[1]) - 1)

    binding = {}
    with open(BASE_DATA_PATH+'/single_mut_effects.csv') as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            pos = float(fields[1]) - 1
            aa_orig = fields[2].strip('"')
            aa_mut = fields[3].strip('"')
            if aa_mut == '*':
                continue
            if fields[8] == 'NA':
                score = float('-inf') #binding average 
            else:
                score = float(fields[8])
            if fields[11] == 'NA':
                expr = float('-inf') #expression average
            else:
                expr = float(fields[11])
            binding[(pos, aa_orig, aa_mut)] = score, expr

    seqs_escape = {}
    with open(BASE_DATA_PATH+'/escape_fracs.csv') as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split(',')
            antibody = fields[2]
            escape_frac = float(fields[10]) #mut_escape_frac_single_mut
            aa_orig = fields[5] #wildtype
            aa_mut = fields[6] #mutation
            pos = int(fields[4]) - 1 #label_site
            assert(seq[pos] == aa_orig)
            escaped = seq[:pos] + aa_mut + seq[pos + 1:]
            assert(len(seq) == len(escaped))
            if escaped not in seqs_escape:
                seqs_escape[escaped] = []
            significant = (
                escape_frac >= survival_cutoff and
                # Statements below should always be true with defaults.
                binding[(pos, aa_orig, aa_mut)][0] >= binding_cutoff and
                binding[(pos, aa_orig, aa_mut)][1] >= expr_cutoff
            )
            seqs_escape[escaped].append({
                'mutant': str(aa_orig)+str(pos) + str(aa_mut),
                'pos': pos,
                'frac_survived': escape_frac,
                'antibody': antibody,
                'significant': significant,
            })

    return seqs_escape
'''
This method loads an additional mutation data from
 escape server resource from tsv file additional_sig_escapes.tsv
'''
def loadAdditionalEscapes():
    AAChange = []
    with open(ADDITIONAL_ESCAPE_SOURCE) as f:
        f.readline() #Consume header
        for line in f:
            fields = line.rstrip().split('\t')
            geneType = fields[0].strip()
            mutation = fields[8].strip()
            if geneType == GENE_TYPE_SPIKE :
                AAChange.append(mutation)

        #print(AAChange[0], "AAChange Length:", len(AAChange))
        return AAChange

'''
This method reads specific mutant csv file such as single residue mutant csv/multiple residue csv 
'''            
def readAdditionalMutants(mutatant_file):
    mutatantFullPath = ADDITIONAL_MUTANT_GENERATED_PATH+SEP+mutatant_file
    mutants = []
    with open(mutatantFullPath, 'r') as f:
        lines = csv.reader(f)
        for line in lines:
            mutant = line[0].strip()
            if mutant is not None:
                mutants.append(mutant)
        #print("Mutants Length:", len(mutants))
        return mutants
def get_significant_mutant_postions():
    mutant_file = MutationTypes.SINGLE_RES_SUB.name+".csv"
    mutants = readAdditionalMutants(mutant_file)
    print("Mutant Lenght: ", len(mutants))
    mutant_positions = []
    for mutant in mutants:
        mutant_pos =   int(mutant[1:-1])
        mutant_positions.append(mutant_pos)
    print(f"Max Position: {max(mutant_positions)} Min Position: {min(mutant_positions)}")
    mutant_positions = mutant_positions + MUL_RES_MUT_POSITIONS + DEL_RES_MUT_POSITIONS + INS_RES_MUT_POSITIONS
    return mutant_positions            

def get_baum_mutants():
    return  set([
        'K417E', 'K444Q', 'V445A', 'N450D', 'Y453F', 'L455F',
        'E484K', 'G485D', 'F486V', 'F490L', 'F490S', 'Q493K',
        'H655Y', 'R682Q', 'R685S', 'V687G', 'G769E', 'Q779K',
        'V1128A'
    ])

def generate_baum_sig_non_sig_csvs():
    seq, seqs_escape = load_baum2020()
    sig_muts = []
    sig_muts_window = []
    non_sig_muts = []
    non_sig_muts_window = []

    for mutated_seq in  seqs_escape:
        mutant = [ data['mutation'] for data  in seqs_escape[mutated_seq]  ][0]
        is_significant = [ data['significant']  for data in seqs_escape[mutated_seq]  ][0]
        if is_significant:
            sig_muts.append({'muts': mutant , 'seqs': mutated_seq })
        else:
            non_sig_muts.append({'muts': mutant , 'seqs': mutated_seq })

    
    df = pd.DataFrame.from_dict(sig_muts)
    df.to_csv('data/gen/baum/baum_sig_muts.csv', index = False, header=True)

    df2 = pd.DataFrame.from_dict(non_sig_muts)
    df2.to_csv('data/gen/baum/baum_non_sig_muts.csv' , index = False, header=True)
'''
baum_non_sig_muts_cleansed.csv File is generated using baum_non_significant_mutants.csv. 
In the cleansed file, the signficant mutatants that are already known/significant 
 (Source: Greany Significant dataset and escape server dataset) are removed from non significant mutant. 
'''
def generate_cleansed_baum_non_signficant_csv():
    #Read baum non sig mut dataset
    df = pd.read_csv('data/additional_escape_variants/gen/baum_non_significant_mutants.csv', header=0)
    non_sig_muts = df.iloc[:, 0].to_list()
    print("Non Sig Muts Before: ", len(non_sig_muts))

    #Read esc sig mut dataset 
    df1 = pd.read_csv('data/additional_escape_variants/gen/SINGLE_RES_SUB.csv', header=0)
    esc_sig_muts = df1.iloc[:, 0].to_list()
    print("ESc Sig Muts Before: ", len(esc_sig_muts))

    #read greany sig mut dataset
    df2 = pd.read_csv('data/additional_escape_variants/gen/greany_significant_mutants.csv', header=0)
    greany_sig_muts = df2.iloc[:, 0].to_list()
    print("Greany Sig Muts Before: ", len(greany_sig_muts))

    #combine greany and esc sig dataset
    combined_sig_muts = esc_sig_muts + greany_sig_muts
    print("combined_sig_muts Length", len(combined_sig_muts))
    unique_combined_sig_muts = list (set(combined_sig_muts))
    print("Duplicate removed combined_sig_muts Length", len(unique_combined_sig_muts))

    #Remove sig data from non sig dataset
    non_sig_after_sig_removed = list (set(non_sig_muts) - set(unique_combined_sig_muts))
    print(f"non_sig lenght Before: {len(non_sig_muts)} and after sig_removed : {len(non_sig_after_sig_removed)}  in baum ")

    #Save Baum Non Sig after removing sig sequences 
    df = pd.DataFrame(non_sig_after_sig_removed)
    df.to_csv('data/gen/baum/baum_non_sig_muts_cleansed.csv', index = False, header=False)




if __name__ == "__main__":
    #readAdditionalMutants("SINGLE_RES_SUB.csv")
    #baum_muts = get_baum_mutants()
    #print(baum_muts)
    #generate_baum_sig_non_sig_csvs()
    #generate_cleansed_baum_non_signficant_csv()
    pass

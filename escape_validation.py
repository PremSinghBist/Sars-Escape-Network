from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from escape import ESCAPE_ANALYSIS_PATH 

def read_analysis_output(file_name):
    filePath = ESCAPE_ANALYSIS_PATH + os.path.sep+file_name
    print("Escape Analysis File Path : ", filePath)
    df = pd.read_csv(filePath , sep='\t', usecols=['Semantic change', 'Grammaticality'])
    semantic_change = df['Semantic change']
    grammaticality = df['Grammaticality']
    return grammaticality.to_list(), semantic_change.to_list()

def compute_spearman_correlation(gram, sem):
    from scipy import stats
    df = pd.DataFrame(
            {'gram': gram, 'sem' : sem }
        )
    #Dropping NA values present in  rows in dataframe
    df = df.dropna()
    #Returns a tuple with 2 values
    corr, p_val = stats.spearmanr(df['gram'].tolist(), df['sem'].tolist())
    print(f"Spearman Correlation result: Semantics vs gram: Corr: {corr} , P-value: {p_val}" )
    return corr, p_val
    




def plot_escape_validataion(grammaticality, semantic_change, grammaticality_sig, semantic_change_sig):
    #fig size original: 4, 6
    fig, ax = plt.subplots(figsize=(5, 5))

    non_sig_seq_length = len(grammaticality) #Plot equal data of sig and non sig

    np.seterr(divide = 'ignore') 
    plt.scatter(grammaticality, np.log10(semantic_change), s=2.8, label="Non Significant")
    plt.scatter(grammaticality_sig[0:non_sig_seq_length], np.log10(semantic_change_sig[0:non_sig_seq_length]), color='r', s=2.8, label='Significant')

    params = {'legend.fontsize': 8, 'legend.scatterpoints':3 }
    plt.rcParams.update(params)
    plt.xlabel("Grammaticality")
    plt.ylabel("Semantic Change", labelpad=1)
    plt.legend()  

    plt.savefig(ESCAPE_ANALYSIS_PATH+"/escape_validation_GISAID_SIG_SAMPLES_output.png")
    plt.close()

def spearman_correlation_wrapper(grammaticality, semantic_change, grammaticality_sig, semantic_change_sig):
    corr, p_val = compute_spearman_correlation(grammaticality, semantic_change)
    corr_sig, p_val_sig = compute_spearman_correlation(grammaticality_sig, semantic_change_sig)
    data = {
            "Correlation": [corr, corr_sig],
            "P-Value" : [p_val, p_val_sig],
        }
    labels = ["Non Significant Sequences", "Significant Sequences"]
    resultDf = pd.DataFrame(
        data, index= labels
    )
    resultDf.to_csv(ESCAPE_ANALYSIS_PATH+"/spearman-correlation-analysis.csv")
    print("Records written to CSV !!!")

if __name__ == "__main__":
    grammaticality, semantic_change = read_analysis_output('gsaid_non_significant_complete__analysis_17000Seq.txt') 
    grammaticality_sig, semantic_change_sig = read_analysis_output('GISAID_SIG_SAMPLES_output.txt') 

    plot_escape_validataion(grammaticality, semantic_change, grammaticality_sig, semantic_change_sig)
    #spearman_correlation_wrapper(compute_spearman_correlation, grammaticality, semantic_change, grammaticality_sig, semantic_change_sig)

    




import SignficantEscapeGenerator as SEG
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import featurizer
import numpy as np
import pandas as pd
import operator
import seaborn as sns
import tensorflow as tf

def write_to_csv(seq_lengths, file_name):
    seq_counter_dict = Counter(seq_lengths)
    import collections
    ordered_dict = collections.OrderedDict(sorted(seq_counter_dict.items(),  key=operator.itemgetter(1),reverse=True))
    records_dict  = {
        'SEQ_LEN':list(ordered_dict.keys()),
        'Freq' : list(ordered_dict.values())
    }
    records = pd.DataFrame(records_dict)
    records.to_csv("data/hist/"+file_name , index=False)
    print(f"File {file_name} writtine successfully")

def plot_histogram(seqs_lengths_freqs, xLabel, yLabel, title,  fig_name):
    #Relative width of bar as fraction of the bin width
    nums, bins, patches = plt.hist(seqs_lengths_freqs, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85) 
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    #plt.text(200, 10500, msg)
    plt.savefig('data/hist/'+fig_name)

def plot_combined_hist(non_sig_seq_lengths, sig_seqs_lengths):
    fig, ax1 = plt.subplots()
    #Relative width of bar as fraction of the bin width
    colors = ['#A52A2A', '#808000' ]
    ax1.hist([sig_seqs_lengths ,non_sig_seq_lengths], color=colors, label=['Significant' , 'Non Significant'], rwidth=4.0) 
    #nums, bins, patches = plt.hist(non_sig_seq_lengths, bins='auto', color='#', alpha=0.7, rwidth=0.85) 
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel("Spike Length")
    plt.ylabel("Frequency")
    plt.title("Significant and Non-significant sequences spike length analysis")
    #plt.text(200, 10500, msg)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig('data/hist/spike_length_hist_analysis.png')

def plot_confusionMatrix(y_test, y_pred, fig_name):
    y_pred = np.argmax(y_pred, axis=1) 
    arry = tf.math.confusion_matrix(y_test, y_pred)
    results = arry/np.sum(arry)
    labels = []
    for result in results:
        for r in result:
            s = str(r) + " %"
            labels.append(s)
    #
    #label_data = np.array(labels).reshape(2, 2)
    ax = sns_plot = sns.heatmap(results, annot=True, fmt=".3%")
    '''for t in ax.texts:
        t.set_text(t.get_text() + " %")'''
    

    ax.set_title("Escape classification Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(['Non Sig', 'Sig'])
    ax.set_yticklabels(['Non Sig', 'Sig'])
    plt.savefig(fig_name)   

def plot_position_heatMap(counter_dict, file_name):
    counter_dict = OrderedDict(counter_dict.most_common())
    dataFrame = pd.DataFrame.from_dict(counter_dict, orient='index', dtype=None)
    ax = sns.heatmap(dataFrame, cmap = "RdBu")
    ax.set_ylabel("Mutant Positions")
    plt.savefig("data/visuals/"+file_name) 
    

def plot_significant_positions():
    from escape import get_significant_mutant_postions
    mutant_positions = get_significant_mutant_postions()
    #Compute Frequency for each position 
    pos_counter_dict = Counter(mutant_positions)
    plot_histogram(mutant_positions, "Position", "Frequency", "Significant Escape Position Analysis",  "significant_escape_position_analysis_histogram.png")
    plot_position_heatMap(pos_counter_dict, 'significant_escape_position_analysis_heat_map.png')

def plot_AUC_curve(y, y_pred, fig_name):
    #RocCurveDisplay.from_predictions(y, y_pred[:,1])
    RocCurveDisplay.from_predictions(y, y_pred)
    plt.savefig(fig_name) 

def plot_Precision_recall_curve(y, y_pred, fig_name):
    PrecisionRecallDisplay.from_predictions(y, y_pred)
    plt.savefig(fig_name) 

def analyze_spike_length():
    non_sig_path =  SEG.getGsaid_non_significant_seq_path()
    non_sig_seqs = SEG.read_sequences(non_sig_path)
    non_sig_seq_lengths = [len(seq) for seq in non_sig_seqs]

    sig_path = SEG.getGsaid_significantSeqs_sample_path()
    sig_seqs =  SEG.read_sequences(sig_path)
    sig_seqs_lengths = [len(sig_seq) for sig_seq in sig_seqs]
    write_to_csv(sig_seqs_lengths, 'sig_freq_counter.csv')
    write_to_csv(non_sig_seq_lengths, 'non_sig_freq_counter.csv')
    xLabel = "Sequence Length"
    yLabel = "Frequency"
    plot_histogram(non_sig_seq_lengths, xLabel, yLabel, "Non Significant Sequences", "non_significant_spike_length_hist_analysis.png")
    plot_histogram(sig_seqs_lengths, xLabel, yLabel, "Significant Sequences", "significant_spike_length_hist_analysis.png")
    plot_combined_hist(non_sig_seq_lengths, sig_seqs_lengths)


if __name__ == "__main__":
    #analyze_spike_length()
    plot_significant_positions()




    pass


    




   
   
    







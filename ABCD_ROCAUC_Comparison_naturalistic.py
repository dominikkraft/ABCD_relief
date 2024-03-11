# Description: This script resembles ABCD_ROCAUC_Comparison_Fig1.py and compares classification performance of different harmonization 
# techniques. We here do no plot the ROC curves, but rather compare the AUC scores for within and across brand comparisons with boxplots. 
# This script results in Figure 2 (FA) in the manuscript and Figure S1 (MD) in the supplementary material.
# Author: Dominik Kraft


import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc, roc_curve
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from os.path import join 
import seaborn as sns



date = "_2024-02-28_0956" #set date that matches the date of the Rscript output files
path = "path/to/data/" #change path to folder that stores Rscript output files


# set up for classification
classifier = QDA()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
label_binarizer = LabelBinarizer()


# set up for plotting
fpr_grid = np.linspace(0.0, 1.0, 1000)
fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
subplot_titles = []
custom_palette = sns.color_palette("Set2", 3)


# number of subjects for each site we exclude later + cumulative sum 
site_subjects = [19, 33, 44, 88, 147, 166, 210, 272, 304]
site_subjects_cumsum = np.cumsum(site_subjects)



# iterate over # sites to be excluded stepwise 
for i in range(0,10):
    
    
    within_dict = {}
    across_dict = {}
    
    if i == 0:
        subplot_titles.append("Full ABCD sample")
    elif i == 1:
        subplot_titles.append("ABCD sample without {} smallest site (n={})".format(i, site_subjects_cumsum[i-1]))
    else:
        subplot_titles.append("ABCD sample without {} smallest sites (n={})".format(i, site_subjects_cumsum[i-1]))
        
    # iterate over different harmonization techniques
    for col_idx,f in enumerate(["ComBat", "CovBat", "RELIEF"]):
            
        #Todo: change md/fa in file name for respective imaging file
        file = "{}_fa_harmonAll_excluded_sites_n={}.csv".format(f.lower(), i)
    
        print(file)
    
        df = pd.read_csv(join(path, file))
        
            
        # rename manufacturer column
        df["manufacturer"] = df["manufacturer"].replace({"DISCOVERY MR750": "GE", "Prisma": "Siemens", "Prisma_fit": "Siemens"})

    
        # select features and convert to numpy array
        X = df.iloc[:, 6:].to_numpy()

        # add manufacturer to site info and set as target 
        y = df.apply(lambda row: "#" + str(row["site"][4:]) + "-" + str(row["manufacturer"]), axis=1)

        
        # perform classification
        y_score = cross_val_predict(classifier, X, y, cv=skf, method='predict_proba')

        # get all possible combinations of labels
        pair_list = list(combinations(np.unique(y), 2))
                
        
        # initialize list to store results 
        pair_scores = []
        mean_tpr = dict()

        # comparison specific mean scores
        siemens_ge = []
        same = []
        
        y_bin = label_binarizer.fit_transform(y) # binarize labels for ROC curve calculation
        
        # iterate over all possible comparisons
        for ix, (label_a, label_b) in enumerate(pair_list):

            # select labels 
            a_mask = y == label_a
            b_mask = y == label_b
            ab_mask = np.logical_or(a_mask, b_mask) 
    
            a_true = a_mask[ab_mask] 
            b_true = b_mask[ab_mask]
            
            idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0] # get idx of respective class with label_a
            idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]
            
            # calculate fpr and tpr for pair comparison, while handling label_a and label_b once as positive class
            fpr_a, tpr_a, _ = roc_curve(a_true, y_score[ab_mask, idx_a])
            fpr_b, tpr_b, _ = roc_curve(b_true, y_score[ab_mask, idx_b])

            # interpolate TPR, average, and calculate AUC
            mean_tpr[ix] = np.zeros_like(fpr_grid)
            mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
            mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
            mean_tpr[ix] /= 2
            mean_score = auc(fpr_grid, mean_tpr[ix])
            pair_scores.append(mean_score)
            
            
            # append comparison specific mean scores to list 
            # Note: this only includes within brand (i.e. S-S, GE-GE) vs across-brand comparisons
            if label_a.split("-")[1] == label_b.split("-")[1]:
                same.append(mean_score) # within 
                    
            elif label_a.split("-")[1] != label_b.split("-")[1]:
                siemens_ge.append(mean_score) # across
            
        within_dict[f] = same
        across_dict[f] = siemens_ge
           

# create dataframe from dictionaries and melt for boxplot
    same_df = pd.DataFrame(within_dict)
    same_df["group"] = "within brands"
    across_df = pd.DataFrame(across_dict)
    across_df["group"] = "across brands"
    
    comp_df = pd.concat([same_df, across_df])

    comp_long = pd.melt(comp_df, id_vars='group', var_name='Method', value_name='AUC')
    
    sns.boxplot(y='Method', x='AUC', hue='group', data=comp_long, 
                ax=axs.flatten()[i], palette=[custom_palette[0], custom_palette[2]], zorder=3)


# after iterative plotting, modify subplots
for i, ax in enumerate(axs.flatten()):
    ax.axvline(0.5, linestyle="--", color="black", zorder=0)
    ax.set_title(subplot_titles[i], fontsize=10)
    ax.set_ylabel("")
    ax.legend(fontsize=9, handlelength=1)
    ax.tick_params(labelbottom=True)


plt.tight_layout()
plt.show()

# save figure

fig.savefig(join(path,"ABCD_Relief_Fig2_FA_revised.png"), dpi=300, bbox_inches="tight")
fig.savefig(join(path,"ABCD_Relief_Fig2_FA_revised.pdf"), bbox_inches="tight")


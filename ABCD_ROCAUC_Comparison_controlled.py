# Description: This script is used to test the classification performance of the three different harmonization methods against unharmonized data.
# The script performs a one-vs-one classification (each ABCD scanner is compared to each other ABCD scanner) and plots the ROC curves for each comparison.
# Individual cuvers are overlayed with the mean curve for within vs across brand comparisons (i.e., siemens-siemens / GE-GE vs siemens-GE). A boxplot
# is added for (harmonization) model performance assessement.
# The script results in Figure 1 in the manuscript.

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
fig, axs = plt.subplots(2, 5, figsize=(24, 10))
subplot_titles = []
custom_palette = sns.color_palette("Set2", 3)


# iterate over modalities and harmonization methods
for row_idx, modality in enumerate(["fa", "md"]):
    
    within_dict = {}
    across_dict = {}
    
    for col_idx,f in enumerate(["raw", "ComBat", "CovBat", "RELIEF"]):
        
        # create titles for subplots
        subplot_titles.append(str(modality.upper()) + " " + str(f))
        
        # last plot in row shows mean curves for each comparison
        if col_idx == 3:
            subplot_titles.append(str(modality.upper()) + " Model comparisons")
        
        # load file 
        file = "{}_{}_n500_{}.csv".format(f, modality, date) 
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
            
    
        # for one-vs-one overall performance
        # ovo_tpr = np.zeros_like(fpr_grid)

        # comparison specific grids for plotting mean per comparison
        siemens_ge_tpr = np.zeros_like(fpr_grid)
        same_tpr = np.zeros_like(fpr_grid)

        # create dict for storing results 
        results_dic = {}

        for ix, (label_a, label_b) in enumerate(pair_list):
            #ovo_tpr += mean_tpr[ix]
            
            # store results in dict
            results_dic[label_a + "-" + label_b] = pair_scores[ix]
            
            # select color depening on comparison made 
            if label_a.split("-")[1] == label_b.split("-")[1]:
                color = custom_palette[0]
                same_tpr += mean_tpr[ix] # within
                
            elif label_a.split("-")[1] != label_b.split("-")[1]:
                color= custom_palette[2]
                siemens_ge_tpr += mean_tpr[ix] # across
            
            # plot individual ROC curves    
            axs[row_idx, col_idx].plot(
                fpr_grid,
                mean_tpr[ix],
                color=color,
                linewidth=1,
                alpha=0.3
                #label=f"Mean {label_a} vs {label_b} (AUC = {pair_scores[ix]:.2f})"     
            ) 

        # plot average comparison specific curves on top of the indivdual curves
        same_tpr /= len(same)
        siemens_ge_tpr /= len(siemens_ge)


        axs[row_idx, col_idx].plot(
            fpr_grid,
            same_tpr,
            label=f"within brands: mean AUC = {np.average(same):.2f}",
            linewidth=2,
            color=custom_palette[0]
        )

        axs[row_idx, col_idx].plot(
            fpr_grid,
            siemens_ge_tpr,
            label=f"across brands: mean AUC = {np.average(siemens_ge):.2f}",
            linewidth=2,
            color=custom_palette[2]
        )
        
        within_dict[f] = same
        across_dict[f] = siemens_ge
        
        # plot mean curves for each comparison in last column
        last_plot_idx = 4
        
        # # plot diagonal line 
        axs[row_idx, col_idx].plot([0, 1], [0, 1], "k--")
        


# combine results for boxplot
    same_df = pd.DataFrame(within_dict)
    same_df["group"] = "within brands"
    across_df = pd.DataFrame(across_dict)
    across_df["group"] = "across brands"
    
    comp_df = pd.concat([same_df, across_df])

    comp_long = pd.melt(comp_df, id_vars='group', var_name='Method', value_name='AUC')
    
    sns.boxplot(y='Method', x='AUC', hue='group', data=comp_long, 
                ax=axs[row_idx, last_plot_idx], palette=[custom_palette[0], custom_palette[2]], zorder=3)


axs[0, last_plot_idx].axvline(0.5, linestyle="--", color="black", zorder=0)
axs[1, last_plot_idx].axvline(0.5, linestyle="--", color="black", zorder=0)

    
#subplot settings
for i, ax in enumerate(axs.flatten()):

    
    ax.legend(fontsize=9.5, loc="lower right", handlelength=1)
    ax.set_title(subplot_titles[i], fontsize=13)
    #ax.axis("square")
    
    if i not in [4, 9]:
        ax.axis("square")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_ylabel("True Positive Rate")

    # Add labels to specific subplots
    if i == 0:
        ax.text(-0.05, 1.1, "A)", transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top')
    elif i == 4:
        ax.text(-0.05, 1.1, "B)", transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top')
        ax.set_ylabel("")
    elif i == 5:
        ax.text(-0.05, 1.1, "C)", transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top')
    elif i == 9:
        ax.text(-0.05, 1.1, "D)", transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top')
        ax.set_ylabel("")

    
plt.tight_layout()
#plt.show()

# save figure

fig.savefig(join(path,"ABCD_Relief_Fig1_revised.png"), dpi=300, bbox_inches="tight")
fig.savefig(join(path,"ABCD_Relief_Fig1_revised.pdf"), bbox_inches="tight")


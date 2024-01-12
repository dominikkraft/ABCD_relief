# Description: This script is used to test the influence of sample size on the classification performance of the three harmonization methods.
# The script is comparable to ABCD_ROCAUC_Comparison_Fig1.py, but additionally iterates over different sample sizes and subsequently plots the results in a bubble plot.
# This scripts results in Figure 2 in the manuscript.

# Author: Dominik Kraft

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
from os.path import join 
from scipy import stats
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelBinarizer



date = "_2023-12-20_1006"
path = "/Users/Dominik/Desktop/relief_test/data/" # change path to folder that stores Rscript output files

# set up for classification
classifier = QDA()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
label_binarizer = LabelBinarizer()


# set up for plotting
fpr_grid = np.linspace(0.0, 1.0, 1000)
custom_palette = sns.color_palette("Set2", 3)
subplot_titles = []

###### figure 2 - bubble plot sample size ######

# initialize lists for bubble plot
mod_col = []
harm_col = []
mean_col_same = []
iqr_col_same = []
mean_col_diff = []
iqr_col_diff = []
sample_col = []


# this for loop section is very similar to the one from ABCD_ROCAUC_Comparison_Fig1.py
# here we additionally iterate over different sample sizes
# check out ABCD_ROCAUC_Comparison_Fig1.py for more detailed comments

for row_idx, modality in enumerate(["fa", "md"]):
    
    for s_idx, sample_size in enumerate(["50", "100", "200", "300", "400", "500"]):
        
        subplot_titles.append(str(modality.upper()) + " (N=" + str(sample_size) + ")")

        for col_idx,f in enumerate(["raw", "ComBat", "RELIEF"]):
            
            file = "{}_{}_n{}_{}.csv".format(f, modality, sample_size, date)
            
            print(file.upper())
            
            df = pd.read_csv(join(path, file))
            
            # rename manufacturer column
            df["manufacturer"] = df["manufacturer"].replace({"DISCOVERY MR750": "GE", "Prisma": "Siemens", "Prisma_fit": "Siemens"})

            X = df.iloc[:, 6:73].to_numpy()

            # add manufacturer to site info
            y = df.apply(lambda row: "#" + str(row["site"][4:]) + "-" + str(row["manufacturer"]), axis=1)

            y_score = cross_val_predict(classifier, X, y, cv=skf, method='predict_proba')

            pair_list = list(combinations(np.unique(y), 2))


            y_bin = label_binarizer.fit_transform(y)
            

            pair_scores = []
            mean_tpr = dict()

            siemens_siemens = []
            siemens_ge = []
            ge_ge = []
            same = []

            for ix, (label_a, label_b) in enumerate(pair_list):

                a_mask = y == label_a
                b_mask = y == label_b
                ab_mask = np.logical_or(a_mask, b_mask) 
                

                a_true = a_mask[ab_mask] 

                b_true = b_mask[ab_mask]
                
                idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]   
                idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]
                
                fpr_a, tpr_a, _ = roc_curve(a_true, y_score[ab_mask, idx_a])
                fpr_b, tpr_b, _ = roc_curve(b_true, y_score[ab_mask, idx_b])

                mean_tpr[ix] = np.zeros_like(fpr_grid)
                mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
                mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
                mean_tpr[ix] /= 2
                mean_score = auc(fpr_grid, mean_tpr[ix])
                pair_scores.append(mean_score)
                
                # append comparison specific mean scores to list 
                if label_a.split("-")[1] == label_b.split("-")[1]:
                    same.append(mean_score)
                    
                        
                elif label_a.split("-")[1] != label_b.split("-")[1]:
                    siemens_ge.append(mean_score)
                
            # comparison specific grids for plotting mean per comparison
            siemens_ge_tpr = np.zeros_like(fpr_grid)
            same_tpr = np.zeros_like(fpr_grid)

            results_dic = {}


            for ix, (label_a, label_b) in enumerate(pair_list):
                #ovo_tpr += mean_tpr[ix]
                
                results_dic[label_a + "-" + label_b] = pair_scores[ix]
                
                # select color depening on comparison made 
                if label_a.split("-")[1] == label_b.split("-")[1]:
                    color = custom_palette[0]
                    same_tpr += mean_tpr[ix]
          
                elif label_a.split("-")[1] != label_b.split("-")[1]:
                    color= custom_palette[2]
                    siemens_ge_tpr += mean_tpr[ix]

            # plot average comparison specific curves
            same_tpr /= len(same)
            siemens_ge_tpr /= len(siemens_ge)
            
            # store results for bubble plot 
            mod_col.append(modality)
            harm_col.append(f)
            sample_col.append(sample_size)
            mean_col_same.append(np.average(same))
            iqr_col_same.append(stats.iqr(same))
            mean_col_diff.append(np.average(siemens_ge))
            iqr_col_diff.append(stats.iqr(siemens_ge))


# create dataframe with filled lists 
df = pd.DataFrame({"modality": mod_col, "Harmonization": harm_col, "N": sample_col, "mean_same": mean_col_same, 
                   "iqr_same": iqr_col_same, "mean_diff": mean_col_diff, "iqr_diff": iqr_col_diff})

# df.to_csv(path + "AUC.csv", index=False) # save if needed

##############################
#### figure 2 bubble plot ####
##############################

# subset df by modality 
df_fa = df.loc[df.modality == "fa"]
df_md = df.loc[df.modality == "md"]

fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(12, 12))

# plot mean AUC vs IQR AUC for within / across brands per modality
s1 = sns.scatterplot(data=df_fa, x="mean_same", y="iqr_same", hue="Harmonization", size="N", sizes=(250, 25), ax=axs[0, 0],
                legend=False, palette="magma", edgecolor="white", linewidth=1)

s2= sns.scatterplot(data=df_md, x="mean_same", y="iqr_same", hue="Harmonization", size="N", sizes=(250, 25), ax=axs[1, 0],
                legend="auto", palette="magma", edgecolor="white", linewidth=1)

s3= sns.scatterplot(data=df_fa, x="mean_diff", y="iqr_diff", hue="Harmonization", size="N", sizes=(250, 25), ax=axs[0, 1],
                legend=False, palette="magma", edgecolor="white", linewidth=1)

s4 = sns.scatterplot(data=df_md, x="mean_diff", y="iqr_diff", hue="Harmonization", size="N", sizes=(250, 25), ax=axs[1, 1],
                legend=False,palette="magma", edgecolor="white", linewidth=1)

titles = ["FA within brands", "FA across brands", "MD within brands", "MD across brands"]


# gradient background
# credit to https://stackoverflow.com/questions/62232420/how-to-achieve-matplotlib-radial-gradient-background

x = np.linspace(0, 1, 256)
y = np.linspace(1, 0, 256)

xArray, yArray = np.meshgrid(x, y)
plotArray = np.sqrt(xArray**2 + yArray**2)

plt.xlim(0.42, 1.05)

for t_idx, ax in enumerate(axs.flatten()):
    ax.imshow(plotArray,
            alpha=0.65,
          cmap=plt.cm.Greys,
          vmin=0,
          vmax=1,
          zorder=-1,
          extent=ax.get_xlim() + ax.get_ylim(),
          aspect='auto')
       
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.set_xlabel("Mean AUC", fontsize=12)
    ax.set_ylabel("IQR AUC", fontsize=12)
    ax.set_title(titles[t_idx], fontsize=14)
    
    
# add legend     
legend = s2.legend(bbox_to_anchor=(1.1, -0.12), loc='upper center', ncol=15, frameon=False, fontsize=12, labelspacing=0.1, handletextpad=-0.2, columnspacing=0.1)
legend.get_texts()[0].set_text("Harmonization:")
legend.get_texts()[4].set_text("N:")

plt.show()

fig.savefig(join(path,"ABCD_Relief_Fig2.png"), dpi=300, bbox_inches="tight")
fig.savefig(join(path,"ABCD_Relief_Fig2.pdf"), bbox_inches="tight")
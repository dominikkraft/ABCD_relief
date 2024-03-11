# Description: This script is used to test whether Combat, Covbat, and Relief Harmonization preserve the biological signal in the data (i.e., signal from the covariates)
# We deploy a simple Machine Learning Framework to predict covariates from raw, Combat, Covbat, and Relief data - and compare the performance of the three methods.
# Result of this script is depicted in Table 1 in the manuscript. Furtheremore, the script outputs the demographic information of the sites.
# This code is based on the controlled harmonization settings (see manuscript for details).

# Author: Dominik Kraft


import pandas as pd
from os.path import join 
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVR, SVC
from tqdm import tqdm


date = "_2024-02-28_0956" #set date that matches the date of the Rscript output files
path = "path/to/data" #change path to folder that stores Rscript output files


def get_demos(file, groupby=False):
    
    global path 
    df = pd.read_csv(join(path, file))
    
    if groupby:
        demos_grouped = df.groupby("site")
        for site, group in demos_grouped:
            print(f"Site: {site}")
            print(f"Number of males in site: {group['sex'].value_counts()[1]}")
            print(f"Number of females in site: {group['sex'].value_counts()[2]}")
            print(f"Mean age in site: {group['age'].mean().round(2)}")
            print(f"Std age in site: {group['age'].std().round(2)}")
            print()
    else:
        print(f"Number of males: {df['sex'].value_counts()[1]}")
        print(f"Number of females: {df['sex'].value_counts()[2]}")
        print(f"Mean age: {df['age'].mean().round(2)}")
        print(f"Std age: {df['age'].std().round(2)}")



### machine learning framework for biological association analyses ###

# set up ML framework to predict covariates 
def bio_ml(df, target="age"):
    # Extract features and target variable
    X = df.iloc[:, 6:].to_numpy()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # depending on target, initialize model and perform cross validation and report performance metric
    if target == "age":
        y = df['age']
        model = SVR()
        predicted = cross_val_predict(model, X, y, cv=5)
        result, _ = pearsonr(y, predicted)
    
    elif target == "sex":
        y = df["sex"]
        model = SVC()
        predicted = cross_val_predict(model, X, y, cv=skf)
        
        result = balanced_accuracy_score(y, predicted)
        
    return result


# perform bioML for single scan sites across modalities and harmonization methods

site_dict = {}

for row_idx, modality in enumerate(["fa", "md"]):
    for col_idx,f in enumerate(["raw", "combat", "covbat", "relief"]):
         
        # load file 
        file = "{}_{}_n500_{}.csv".format(f, modality, date) 
        print(file)
    
        df = pd.read_csv(join(path, file))
            
        # rename manufacturer column
        df["manufacturer"] = df["manufacturer"].replace({"DISCOVERY MR750": "GE", "Prisma": "Siemens", "Prisma_fit": "Siemens"})

        # add manufacturer to site info and set as target 
        df["site"] = df.apply(lambda row: "#" + str(row["site"][4:]) + "-" + str(row["manufacturer"]), axis=1)
        
         # iterate over targets and perform bioML
        for target in ["age", "sex"]:  
            for site in df["site"].unique(): # iterate over sites
                
                site_df = df[df["site"] == site]
                
                site_df = site_df.sample(n=496, random_state=42) # sample 496 subjects to match the procedure for mixture of sites
                
                res = bio_ml(site_df, target=target)
                
                key = "{}_{}_{}".format(modality, f, target)
                site_dict.setdefault(site, []).append({key: res})


# transform nested dictionary into list of dictionaries -> dataframe
data = [
    {"site": site, **{key: result for result_dict in results_list for key, result in result_dict.items()}}
    for site, results_list in site_dict.items()
]
single_site_ml = pd.DataFrame(data)


# perform bioML for mixture of single scan sites across modalities and harmonization methods
# for the mixture we sample 62 subjects from each of the 8 sites 500 times 

mix_dict = {}

for row_idx, modality in enumerate(["fa", "md"]):
    for col_idx,f in enumerate(["raw", "combat", "covbat", "relief"]):
        
           
        # load file 
        file = "{}_{}_n500_{}.csv".format(f, modality, date) 
        print(file)
        df = pd.read_csv(join(path, file))
            
        # rename manufacturer column
        df["manufacturer"] = df["manufacturer"].replace({"DISCOVERY MR750": "GE", "Prisma": "Siemens", "Prisma_fit": "Siemens"})
        # add manufacturer to site info and set as target 
        df["site"] = df.apply(lambda row: "#" + str(row["site"][4:]) + "-" + str(row["manufacturer"]), axis=1)
        
        # iterate over targets and perform bioML
        for target in ["age", "sex"]:
            
            key = "{}_{}_{}".format(modality, f, target)
            
            results = []

            # iterate over 500 seeds
            for seed in tqdm(range(500), desc="Processing", unit="iteration"):
                
                sampled_df = df.groupby("site", group_keys=False, sort=False).apply(lambda group: group.sample(62, replace=False, random_state=seed))
            
                # Reset the index of the final DataFrame
                sampled_df.reset_index(drop=True, inplace=True)
                
                result = (bio_ml(sampled_df, target=target))
                results.append(result)
            
            
            mix_dict[key] = results


mix_site_ml = pd.DataFrame(mix_dict)


# Calculate mean and standard deviation for numeric columns in single_site_ml
single_site_ml_numeric = single_site_ml.select_dtypes(include=[float, int])
single_site_ml_mean = single_site_ml_numeric.mean().round(3)
single_site_ml_std = single_site_ml_numeric.std().round(3)

# Print column name, mean, and standard deviation for single_site_ml
print("SINGLE")
for column in single_site_ml_mean.index:
    print(f"Column: {column}, Mean: {single_site_ml_mean[column]}, Std: {single_site_ml_std[column]}")

# Calculate mean and standard deviation for numeric columns in mix_site_ml
mix_site_ml_numeric = mix_site_ml.select_dtypes(include=[float, int])
mix_site_ml_mean = mix_site_ml_numeric.mean().round(3)
mix_site_ml_std = mix_site_ml_numeric.std().round(3)

# Print column name, mean, and standard deviation for mix_site_ml
print("MIXTURE")
for column in mix_site_ml_mean.index:
    print(f"Column: {column}, Mean: {mix_site_ml_mean[column]}, Std: {mix_site_ml_std[column]}")



print(f"DEMOS from {file}")

get_demos(file, groupby=True)

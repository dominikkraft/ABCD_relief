# Author: Dominik Kraft 
# This code performs data loading and handling of the ABCD data and subsequently performs 
# Combat, Covbat, and Relief harmonization and saves raw/combat/covbat/relief data files for .py scripts
# This code outputs results from the "naturalistic setting" - i.e., harmonization is performed in the whole ABCD sample
# see manuscript for additional information.

rm(list = ls())
set.seed(123)

# set datapath 
DATAPATH = "path_to/abcd-data-release-5.0/core" 

# set subjectpath
SUBPATH = "path_to_output/controlledharmonization" #set to ABCD_Harmonization_controlled.R output path


date= "_2024-02-28_0956" # get information from ABCD_Harmonization_controlled.R output

# import packages 
library(neuroCombat)
library(RELIEF)
library(CovBat)
library(data.table)
library(dplyr)
library(tidyr)


# loading reference subjects for n=500 (modality/harmonization not relevant)
ref_subs <- fread(file.path(SUBPATH, paste0("raw_md_n500_", date, ".csv")), data.table = F)

# loading demographics 
demo1<-fread(file.path(DATAPATH, "abcd-general/abcd_p_demo.csv"), data.table = F)
demo2<-fread(file.path(DATAPATH, "abcd-general/abcd_y_lt.csv"), data.table = F)
demo3<-fread(file.path(DATAPATH, "imaging/mri_y_adm_info.csv"), data.table = F)


#select subject, age, sex and scanner site / manufacturer 
demo1<-demo1[, c(1:2, 9)]
demo2<-demo2[, c(1:2, 9)]
demo3<-demo3[, c(1:2, 5:6)] 


#merge demos 
demo<-Reduce(function(x,y) merge(x, y, by= c("src_subject_id", "eventname")),
               list( demo1, demo2, demo3))


# perform some data wrangling, re-naming etc. on demo 
demo<-rename(demo, subject = src_subject_id)
demo$eventname <- gsub("(?<=^|_)([a-z])", "\\U\\1", demo$eventname, perl=TRUE)
demo$eventname<- gsub("_", "", demo$eventname)
demo$subject<- gsub("_", "", demo$subject)
demo<-rename(demo, session = eventname)
demo<-rename(demo, age = interview_age)
demo<-rename(demo,sex = demo_sex_v2)
demo<-rename(demo, site = mri_info_deviceserialnumber)
demo<-rename(demo, manufacturer = mri_info_manufacturersmn)

demo<- demo %>%
  group_by(subject) %>% 
  fill(sex, .direction = "downup")  %>%
  ungroup

## select only baseline data 
baseline <- demo[demo$session == "BaselineYear1Arm1", ]

# select only male / female data
baseline <- baseline[baseline$sex %in% list(1,2),]


# define function for loading and processing imaging data 
process_data <- function(file_path) {
  # Load data
  data <- fread(file_path, data.table = FALSE)
  
  # Clean up eventname column
  data$eventname <- gsub("(?<=^|_)([a-z])", "\\U\\1", data$eventname, perl=TRUE)
  data$eventname <- gsub("_", "", data$eventname)
  
  # Rename columns
  data <- rename(data, subject = src_subject_id)
  data$subject <- gsub("_", "", data$subject)
  data <- rename(data, session = eventname)
  
  # Filter rows for the desired session
  data <- data[data$session == "BaselineYear1Arm1", ]
  
  # Ignore columns with "all" for fa and md
  data <- data[, !grepl("all", names(data))]
  
  # Omit rows with NaN values
  data <- na.omit(data)
  
  return(data)
}

# fa = fractional anisotropy, md = mean diffusivity 
fa <- process_data(file.path(DATAPATH, "imaging/mri_y_dti_fa_is_at.csv"))
md <- process_data(file.path(DATAPATH, "imaging/mri_y_dti_md_is_at.csv"))



## merge demo and brain dataframes (fa, md) 
df1<-Reduce(function(x,y) merge(x, y, by= c("subject", "session")),
               list(baseline,fa))
df2<-Reduce(function(x,y) merge(x, y, by= c("subject", "session")),
            list(baseline,md))



# define harmonization function
harmonization_function <- function(data_frame, output_prefix, size, write_output=TRUE) {
  
  # Convert 'sex' column to factor
  #data_frame$sex <- as.factor(data_frame$sex)
  
  # Select covariates
  covar <- data_frame[, c(1:6)]
  
  # Extract 'site' for batch
  batch <- covar$site
  
  # Extract data for fa/md
  data <- data_frame[, 7:43]

  # Transform data
  data <- t(data)
  
  # Define covariates
  modcovar <- model.matrix(~ as.factor(sex) + age, data = covar)
  
  # Combat harmonization
  combat_result <- neuroCombat(dat = data, batch = batch, mod = modcovar)
  datcombat <- t(as.data.frame(combat_result$dat.combat))
  combat_harmonized <- cbind(covar, datcombat)
  
  # Relief harmonization
  relief_result <- relief(dat = data, batch = batch, mod = modcovar)
  datrelief <- t(as.data.frame(relief_result$dat.relief))
  relief_harmonized <- cbind(covar, datrelief)
  
  # CovBat harmonization
  covbat_result <- covbat(dat = data, bat = batch, mod = modcovar)
  datcovbat <- t(as.data.frame(covbat_result$dat.covbat))
  covbat_harmonized <- cbind(covar, datcovbat)
  
  result_list <- list(
    combat = combat_harmonized,
    covbat = covbat_harmonized,
    relief = relief_harmonized)
  
  
  # Write tables
  
  if (write_output) {
    write.table(data_frame, file = paste0("raw_", output_prefix, "_n", size, "_", date, ".csv"), sep = ",", row.names = FALSE)
    write.table(combat_harmonized, file = paste0("combat_", output_prefix,"_n", size, "_", date, ".csv"), sep = ",", row.names = FALSE)
    write.table(relief_harmonized, file = paste0("relief_", output_prefix, "_n", size, "_",date, ".csv"), sep = ",", row.names = FALSE)
    write.table(covbat_harmonized, file = paste0("covbat_", output_prefix, "_n", size, "_",date, ".csv"), sep = ",", row.names = FALSE)
  }
  
  return(result_list)

}

# stepwise exclusion of i-smallest scanning sites
# note that this code does not return full sample without excluded sites

for (i in 1:9) {
  cat("----- deleting", i, "sites ----")
  
  print(sort(table(df1$site))[1:i])
  
  excl <- names(sort(table(df1$site))[1:i])
  
  df1_temp <- df1[!(df1$site %in% excl), ]
  df2_temp <- df2[!(df2$site %in% excl), ]
  
  stopifnot(identical(sort(df1_temp$subject), sort(df2_temp$subject)))
  
  # perform harmonization without subsampling, i.e., on all sites and subjects at baseline
  fa_result_real = harmonization_function(df1_temp, "fa", nrow(df1_temp), write_output = FALSE)
  
  # use reference subjects for benchmarking (i.e., 4000 subjects that were used in simulated scenario)
  # FA 
  fa_result_real500 <- lapply(fa_result_real, function(df) {
    df_filtered <- df %>%
      filter(subject %in% ref_subs$subject) %>%
      arrange(subject)
    
    stopifnot(identical(df_filtered$subject, ref_subs$subject))
    return(df_filtered)
  })
  
  # save FA
  lapply(names(fa_result_real500), function(name) {
    write.table(fa_result_real500[[name]], file = paste0(name, "_fa_harmonAll_excluded_sites_n=", i, ".csv"), sep = ",", row.names = FALSE)
  })
  
  
  # MD
  md_result_real = harmonization_function(df2_temp, "md", nrow(df2_temp), write_output = FALSE)
  
  
  # choose the same 500 subjects and sites as done above
  md_result_real500 <- lapply(md_result_real, function(df) {
    df_filtered <- df %>%
      filter(subject %in% ref_subs$subject) %>%
      arrange(subject)
    
    stopifnot(identical(df_filtered$subject, ref_subs$subject))
    return(df_filtered)
  })
  
  # save
  lapply(names(md_result_real500), function(name) {
    write.table(md_result_real500[[name]], file = paste0(name, "_md_harmonAll_excluded_sites_n=", i, ".csv"), sep = ",", row.names = FALSE)
  })
  

}








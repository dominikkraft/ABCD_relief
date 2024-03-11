# Author: Dominik Kraft 
# This code performs data loading and handling of the ABCD data and subsequently performs 
# Combat, Covbat, and Relief harmonization and saves raw/combat/covbat/relief data files for .py scripts
# This code outputs results from the "controlled setting" - see manuscript for additional information. 


rm(list = ls())


# set datapath !
DATAPATH = "/path_to/abcd-data-release-5.0/core" 


set.seed(123)
date=format(Sys.time(), "_%Y-%m-%d_%H%M")

# import packages 
library(neuroCombat)
library(RELIEF)
library(CovBat)
library(data.table)
library(dplyr)
library(tidyr)


# loading demographics 
demo1<-fread(file.path(DATAPATH, "abcd-general/abcd_p_demo.csv"), data.table = F)
demo2<-fread(file.path(DATAPATH, "abcd-general/abcd_y_lt.csv"), data.table = F)
demo3<-fread(file.path(DATAPATH, "imaging/mri_y_adm_info.csv"), data.table = F)


#select subject, age, sex and scanner site / manufacturer 
demo1<-demo1[, c(1:2, 9)]
demo2<-demo2[, c(1:2, 9)]
demo3<-demo3[, c(1:2, 5:6)] 


# choose only scanners from SIEMENS (prisma,fit) and GE (Discovery) for simulation part 
demo3_filtered <- demo3[demo3$mri_info_manufacturersmn %in% list("Prisma", "Prisma_fit", "DISCOVERY MR750"),]

#merge demos 
demo<-Reduce(function(x,y) merge(x, y, by= c("src_subject_id", "eventname")),
               list( demo1, demo2, demo3_filtered))


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


# SUBSAMPLING

# 1) select sites with >= 500 subjects
df1_filtered <- subset(df1, table(site)[as.character(site)] > 500)
df2_filtered <- subset(df2, table(site)[as.character(site)] > 500)


# select n=500 subjects from each site (some sites contain > 500 subjects)

# sample from df1
df1_sampled  <- df1_filtered %>%
  group_split(site) %>%
  lapply(function(group) sample_n(group, size = 500, replace=FALSE)) %>%
  bind_rows() %>%
  arrange(subject)

# choose same subjects from df2
df2_sampled <- df2_filtered %>%
  filter(subject %in% df1_sampled$subject) %>% 
  arrange(subject)

stopifnot(identical(df1_sampled$subject, df2_sampled$subject))




# define function that samples different sample sizes from dataframes 
# note: all sampling is always done per scanning site / group 
sampling <- function(df1, df2, sizes) {
  result_list <- list()
  
  for (size in sizes) {
    # Sample from df1
    df1_sampled <- df1 %>%
      group_split(site) %>%
      lapply(function(group) sample_n(group, size = size, replace = FALSE)) %>%
      bind_rows() %>%
      arrange(subject)
    
    # choose same subjects from df2
    df2_sampled <- df2 %>%
      filter(subject %in% df1_sampled$subject) %>%
      arrange(subject)
    
    # Check if subjects are identical
    stopifnot(identical(df1_sampled$subject, df2_sampled$subject))
    
    # Store the results in a list
    result_list[[paste0("n_", size)]] <- list(df1_sampled = df1_sampled, df2_sampled = df2_sampled)
  }
  
  return(result_list)
}

# Specify sample sizes 
sample_sizes <- c(50, 100, 200, 300, 400, 500)



# perform sampling
result <- sampling(df1_sampled, df2_sampled, sample_sizes)



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

# perform harmonization and save files 
for (si in sample_sizes) {
  print(si)
  # Access df1 and df2 for the current sample size
  current_result <- result[[paste0("n_", si)]]
  
  fa_sampled <- current_result$df1_sampled
  md_sampled <- current_result$df2_sampled

  fa_result_sim <- harmonization_function(fa_sampled, "fa", si, write_output = TRUE)
  md_result_sim <- harmonization_function(md_sampled, "md", si, write_output = TRUE)
}
print(date) # get date for filename








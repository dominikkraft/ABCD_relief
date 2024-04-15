# ABCD_relief

This repository stores code and analyses for our recent commentary in ImagingNeuroscience entitled 

> Removing scanner effects with a multivariate latent approach - a RELIEF for the ABCD imaging data?

[Read the paper here!](https://doi.org/10.1162/imag_a_00157)


* This works builds upon Zhang et al (2023) and tests RELIEF´s performance in the ABCD study. 


This code was performed in RStudio (R version 4.2.3) and python (version 3.9.16). 

The following main packages were used
- `neuroCombat version 1.0.13 in R` see ![NeuroCombat](https://github.com/Jfortin1/neuroCombat_Rpackage)
- `RELIEF  version 0.1.0 in R` see ![RELIEF](https://github.com/junjypark/RELIEF)
- `CovBat version 0.1.0 in R` see ![CovBat](https://github.com/andy1764/CovBat_Harmonization)
- `skicit-learn version 1.3.2. in python`


## Order of Operations

- `ABCD_Harmonization_.R` performs data loading, handling and harmonization procedure with ComBat and RELIEF - we perform the harmonization in a controlled and naturalistic setting 
- `ABCD_ROCAUC_Comparison_Fig1.py` investigates scanner classification performance from (un)-harmonized data - comparisons are in controlled / naturalistic setting
- `ABCD_SampleInflue_controlled.py` investigates sample size influence on harmonization performance - only in controlled setting
- `ABCD_BioML_Table1.py` investigates the harmonization technique´s ability to retain signal related to covariates + provides demographics



## Reference 

> Zhang, R., Oliver, L. D., Voineskos, A. N., & Park, J. Y. (2023). RELIEF: A structured multivariate approach for removal of latent inter-scanner effects. Imaging Neuroscience (Cambridge, Mass.), 1, 1–16. https://doi.org/10.1162/imag_a_00011

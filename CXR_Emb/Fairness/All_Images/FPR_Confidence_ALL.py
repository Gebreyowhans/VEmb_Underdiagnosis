import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mean

import warnings
import sys

def main():
  
  sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
  sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "MIMIC_CXR_EMB")))
  
  from MIMIC_CXR_EMB.config_MIMIC import get_utility_variables
  from MIMIC_CXR_EMB.FPR_Confidence_MIMIC import get_avg_subgroup__sex_fpr,get_avg_subgroup__age_fpr,get_avg_subgroup__race_fpr,get_avg_AgeSex_fpr,get_avg_RaceSex_fpr,get_avg_RaceAge_fpr,get_avg_RaceAgeSex_fpr
    
  
  # Get utility variables
  utility_variables = get_utility_variables()
  number_of_runs=utility_variables['number_of_runs']
  significance_level = utility_variables['significance_level'] 
  
  # Create directory for FPR
  subgroup_FPR = os.path.join('./FPR/SubGroup_FPR')

  # subgroup average FPR
  get_avg_subgroup__sex_fpr(subgroup_FPR,significance_level)
  get_avg_subgroup__age_fpr(subgroup_FPR,significance_level)
  get_avg_subgroup__race_fpr(subgroup_FPR,significance_level)
 
  
  
  # Two group average FPR
  
  twogroup_FPR="./FPR/Two_Group_Intersection_FPR"
  
  get_avg_AgeSex_fpr(twogroup_FPR,significance_level)
  get_avg_RaceSex_fpr(twogroup_FPR,significance_level)
  get_avg_RaceAge_fpr(twogroup_FPR,significance_level)

  
  # Three group average FPR
  three_group_dir = './FPR/Three_Group_Intersection_FPR'
  get_avg_RaceAgeSex_fpr(three_group_dir,significance_level)


if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  main()
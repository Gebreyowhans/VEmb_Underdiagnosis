import pandas as pd
import numpy as np
import math
import random as python_random
import io
import os
import glob
import sys
import os
import warnings

def main():
  
  
  sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
  sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "MIMIC_CXR_EMB")))

  from MIMIC_CXR_EMB.MIMIC_FPR import Subgroup_FPR, Two_Group_FPR, Three_Group_FPR
  from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups, get_utility_variables
  #from MIMIC_CXR_EMB.FPR_Disparities_MIMIC_CXR import get_Sex_FPR_Disparities, get_Age_FPR_Disparities, get_Race_FPR_Disparities
  from MIMIC_CXR_EMB.config_MIMIC import get_diseases, get_diseases_abbr, get_seeds, get_patient_groups
  
  diseases = get_diseases()
  diseases_abbr = get_diseases_abbr()

  patient_groups = get_patient_groups()

  gender = patient_groups["sex"]
  age_decile = patient_groups["age"]
  race = patient_groups["race"]

  factor = [gender, age_decile, race]

  factor_str = ['gender', 'age_decile', 'race']

  seeds = get_seeds()
  
  
  for seed in seeds:
    
    np.random.seed(seed)
    python_random.seed(seed)
    prediction_results_path = "./Prediction_results/"
    
    fpr_npr_path = "./FPR/SubGroup_FPR/"
    os.makedirs(os.path.dirname(fpr_npr_path), exist_ok=True)
    df = pd.read_csv(f"{prediction_results_path}bipred_{seed}.csv")
    
    
    '''' FPR Calculation '''
    
    for i in range(len(factor)):
        Subgroup_FPR(df,diseases,factor[i],factor_str[i],seed,fpr_npr_path)

    '''' Two group FPR '''
    fpr_npr_path_2_group_intersection = "./FPR/Two_Group_Intersection_FPR/"
    os.makedirs(os.path.dirname(fpr_npr_path_2_group_intersection), exist_ok=True)
    
    Two_Group_FPR(df, diseases, gender, 'gender',race,'race',seed,fpr_npr_path_2_group_intersection)
    Two_Group_FPR(df, diseases, gender, 'gender', age_decile, 'age_decile',seed,fpr_npr_path_2_group_intersection)
    Two_Group_FPR(df, diseases, race, 'race', age_decile, 'age_decile',seed,fpr_npr_path_2_group_intersection)

    '''' Three group FPR '''
    fpr_npr_path_3_group_intersection = "./FPR/Three_Group_Intersection_FPR/"
    os.makedirs(os.path.dirname(fpr_npr_path_3_group_intersection), exist_ok=True)

    Three_Group_FPR(df, diseases,gender,'gender',age_decile, 
                    'age_decile',race,'race',
                    seed,fpr_npr_path_3_group_intersection)

    print("Seed : ",seed)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()



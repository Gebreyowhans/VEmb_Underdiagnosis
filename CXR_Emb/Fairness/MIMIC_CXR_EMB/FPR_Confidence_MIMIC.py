import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mean

import warnings
import sys
from config_MIMIC import get_utility_variables

def FiveRunSubgroup(factors, df_in, df_out,significance_level=1.96):
    fpr = []
    fnr = []
    percent = []
    ci_fpr =[]
    ci_fnr =[]
    confI = significance_level * df_in.loc['std'] / np.sqrt(5)

    for fact in factors:
        percent.append(round(df_in.loc['mean']['#'+fact],3))
        fpr.append(round(df_in.loc['mean']['FPR_'+fact],3))
        ci_fpr.append(round(confI.loc['FPR_'+fact],3))
      

    df_out['#'] = percent
    df_out['FPR']=fpr
    df_out['CI_FPR']=ci_fpr

   

    return df_out

def FiveRun(factors,output_df,df,significance_level=1.96):
  for factor in factors:
    fpr = round(df[factor]['mean'],3)
    confI = round(significance_level * df[factor]["std"] / np.sqrt(5),3)
    output_df[factor] = pd.DataFrame(fpr.values.tolist(),columns =[factor])
    output_df['CI_'+factor] = pd.DataFrame(confI.values.tolist(),columns =['CI_'+factor])

  return output_df


def get_avg_subgroup__sex_fpr(path,significance_level=1.96):
  
  #subgroup_FPR = os.path.join('./FPR/SubGroup_FPR')
  subgroup_FPR = path
  
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FPR_FNR_NF_sex.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  seed_19_sex = pd.read_csv(subgroup_FPR+"/run_19FPR_FNR_NF_sex.csv",index_col=0)
  seed_31_sex = pd.read_csv(subgroup_FPR+"/run_31FPR_FNR_NF_sex.csv",index_col=0)
  seed_38_sex = pd.read_csv(subgroup_FPR+"/run_38FPR_FNR_NF_sex.csv",index_col=0)
  seed_47_sex = pd.read_csv(subgroup_FPR+"/run_47FPR_FNR_NF_sex.csv",index_col=0)
  seed_77_sex = pd.read_csv(subgroup_FPR+"/run_77FPR_FNR_NF_sex.csv",index_col=0)
  

  sex_dataframes = [seed_19_sex, seed_31_sex, seed_38_sex, seed_47_sex, seed_77_sex]
  result_sex = pd.concat(sex_dataframes)

  result_sex_df = result_sex.describe()

  sex = ['M','F' ]
  fpr_fnr_sex_df = pd.DataFrame(sex, columns=["sex"])

  # get average sex FPR over 5 runs with 95% confidence interval
  fpr_fnr_sex_df=FiveRunSubgroup(sex, result_sex_df, fpr_fnr_sex_df,significance_level)
  fpr_fnr_sex_df.to_csv(subgroup_FPR+'/Subgroup_FNR_FPR_Sex.csv',index=False)
  

def get_avg_subgroup__age_fpr(path,significance_level=1.96):
  
  subgroup_FPR = path
  
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FPR_FNR_NF_age.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  seed_19_age = pd.read_csv(subgroup_FPR+"/run_19FPR_FNR_NF_age.csv",index_col=0)
  seed_31_age = pd.read_csv(subgroup_FPR+"/run_31FPR_FNR_NF_age.csv",index_col=0)
  seed_38_age = pd.read_csv(subgroup_FPR+"/run_38FPR_FNR_NF_age.csv",index_col=0)
  seed_47_age = pd.read_csv(subgroup_FPR+"/run_47FPR_FNR_NF_age.csv",index_col=0)
  seed_77_age = pd.read_csv(subgroup_FPR+"/run_77FPR_FNR_NF_age.csv",index_col=0)
  

  age_dataframes = [seed_19_age, seed_31_age, seed_38_age, seed_47_age, seed_77_age]
  result_age_df= pd.concat(age_dataframes)

  result_age_df = result_age_df.describe()

  age = ['0-20','20-40','40-60','60-80','80+' ]
  fpr_fnr_age_df = pd.DataFrame(age, columns=["Age"])

  # get average age FPR over 5 runs with 95% confidence interval
  fpr_fnr_age_df=FiveRunSubgroup(age, result_age_df, fpr_fnr_age_df,significance_level)
  fpr_fnr_age_df.to_csv(subgroup_FPR+'/Subgrounp_FNR_FPR_Age.csv',index=False)


def get_avg_subgroup__race_fpr(path,significance_level=1.96):
  
  subgroup_FPR = path
  
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FPR_FNR_NF_race.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  seed_19_race = pd.read_csv(subgroup_FPR+"/run_19FPR_FNR_NF_race.csv",index_col=0)
  seed_31_race = pd.read_csv(subgroup_FPR+"/run_31FPR_FNR_NF_race.csv",index_col=0)
  seed_38_race = pd.read_csv(subgroup_FPR+"/run_38FPR_FNR_NF_race.csv",index_col=0)
  seed_47_race = pd.read_csv(subgroup_FPR+"/run_47FPR_FNR_NF_race.csv",index_col=0)
  seed_77_race = pd.read_csv(subgroup_FPR+"/run_77FPR_FNR_NF_race.csv",index_col=0)
  seed_77_race.head(3)

  race_dataframes = [seed_19_race, seed_31_race, seed_38_race, seed_47_race, seed_77_race]
  result_race= pd.concat(race_dataframes)
  result_race_df =result_race.describe()


  race = ['White','Black','Hisp','Other','Asian','American' ]
  fpr_fpr_race_df = pd.DataFrame(race, columns=["Race"])

  # get average race FPR over 5 runs with 95% confidence interval
  fpr_fpr_race_df=FiveRunSubgroup(race, result_race_df, fpr_fpr_race_df,significance_level)
  fpr_fpr_race_df.to_csv(subgroup_FPR+'/Subgroup_FNR_FPR_Race.csv',index=False)


def get_avg_subgroup_insurance_fpr(path,significance_level=1.96):
  
  subgroup_FPR = path
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FPR_FNR_NF_insurance.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  seed_19_insu = pd.read_csv(subgroup_FPR+"/run_19FPR_FNR_NF_insurance.csv",index_col=0)
  seed_31_insu = pd.read_csv(subgroup_FPR+"/run_31FPR_FNR_NF_insurance.csv",index_col=0)
  seed_38_insu = pd.read_csv(subgroup_FPR+"/run_38FPR_FNR_NF_insurance.csv",index_col=0)
  seed_47_insu = pd.read_csv(subgroup_FPR+"/run_47FPR_FNR_NF_insurance.csv",index_col=0)
  seed_77_insu = pd.read_csv(subgroup_FPR+"/run_77FPR_FNR_NF_insurance.csv",index_col=0)
  
  insu_dataframes = [seed_19_insu, seed_31_insu, seed_38_insu, seed_47_insu, seed_77_insu]
  result_insu= pd.concat(insu_dataframes)

  result_insu_df =result_insu.describe()

  insurance = ['Medicare','Other','Medicaid' ]
  fpr_fpr_insu_df = pd.DataFrame(insurance, columns=["Insurance"])
  
  # get average race FPR over 5 runs with 95% confidence interval
  fpr_fpr_insu_df=FiveRunSubgroup(insurance, result_insu_df,fpr_fpr_insu_df,significance_level)
  fpr_fpr_insu_df.to_csv(subgroup_FPR+'/Subgroup_FNR_FPR_Insu.csv',index=False)

def get_avg_AgeSex_fpr(path,significance_level=1.96):
  
  #twogroup_FPR="./FPR/Two_Group_Intersection_FPR"
  twogroup_FPR=path
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FP_AgeSex.csv 
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  seed_19_agesex = pd.read_csv(twogroup_FPR+"/run_19FP_AgeSex.csv")
  seed_31_agesex= pd.read_csv(twogroup_FPR+"/run_31FP_AgeSex.csv")
  seed_38_agesex = pd.read_csv(twogroup_FPR+"/run_38FP_AgeSex.csv")
  seed_47_agesex = pd.read_csv(twogroup_FPR+"/run_47FP_AgeSex.csv")
  seed_77_agesex = pd.read_csv(twogroup_FPR+"/run_77FP_AgeSex.csv")

  fp_agesex =pd.concat([seed_19_agesex, seed_31_agesex,seed_38_agesex, seed_47_agesex,seed_77_agesex])
  fp_agesex =fp_agesex.groupby("Age")
  fp_agesex = fp_agesex.describe()


  factors = ['FPR_F', 'FPR_M']
  age =['0-20', '20-40', '40-60', '60-80','80-']
  agesex_df = pd.DataFrame(age, columns=["Age"])

  agesex_df = FiveRun(factors,agesex_df,fp_agesex,significance_level)
  agesex_df.to_csv(twogroup_FPR+'/Inter_AgeSex.csv',index=False)

def get_avg_RaceSex_fpr(path,significance_level=1.96):
  
  twogroup_FPR=path
  seed_19_race_sex = pd.read_csv(twogroup_FPR+"/run_19FP_RaceSex.csv")
  seed_31_race_sex= pd.read_csv(twogroup_FPR+"/run_31FP_RaceSex.csv")
  seed_38_race_sex = pd.read_csv(twogroup_FPR+"/run_38FP_RaceSex.csv")
  seed_47_race_sex = pd.read_csv(twogroup_FPR+"/run_47FP_RaceSex.csv")
  seed_77_race_sex = pd.read_csv(twogroup_FPR+"/run_77FP_RaceSex.csv")

  fp_race_sex =pd.concat([seed_19_race_sex, seed_31_race_sex,seed_38_race_sex,
                        seed_47_race_sex,seed_77_race_sex])

  fp_race_sex =fp_race_sex.groupby("race")
  fp_race_sex = fp_race_sex.describe()

  factors = ['FPR_F', 'FPR_M']
  race =['AMERICAN INDIAN/ALASKA NATIVE', 'ASIAN', 'BLACK/AFRICAN AMERICAN',
        'HISPANIC/LATINO','OTHER','WHITE']
  RaceSex_df = pd.DataFrame(race, columns=["race"])

  RaceSex_df = FiveRun(factors,RaceSex_df,fp_race_sex,significance_level)
  RaceSex_df.to_csv(twogroup_FPR+'/Inter_RaceSex.csv',index=False)

def get_avg_RaceAge_fpr(path,significance_level=1.96):
  
  twogroup_FPR=path
  
  seed_19_race_age = pd.read_csv(twogroup_FPR+"/run_19FP_RaceAge.csv")
  seed_31_race_age= pd.read_csv(twogroup_FPR+"/run_31FP_RaceAge.csv")
  seed_38_race_age = pd.read_csv(twogroup_FPR+"/run_38FP_RaceAge.csv")
  seed_47_race_age = pd.read_csv(twogroup_FPR+"/run_47FP_RaceAge.csv")
  seed_77_race_age = pd.read_csv(twogroup_FPR+"/run_77FP_RaceAge.csv")

  fp_race_age =pd.concat([seed_19_race_age, seed_31_race_age,seed_38_race_age,
                        seed_47_race_age,seed_77_race_age])

  fp_race_age =fp_race_age.groupby("age")
  fp_race_age = fp_race_age.describe()


  factors = ['FPR_White', 'FPR_Black','FPR_Hisp','FPR_Other','FPR_Asian','FPR_American']
  age =['0-20', '20-40', '40-60', '60-80','80-']
  RaceAge_df = pd.DataFrame(age, columns=["age"])


  RaceAge_df = FiveRun(factors,RaceAge_df,fp_race_age,significance_level)
  RaceAge_df.to_csv(twogroup_FPR+'/Inter_RaceAge.csv',index=False)


def get_avg_InsSex_fpr(path,significance_level=1.96):
  
  twogroup_FPR=path
  
  
  FP5_InsSex = pd.read_csv(twogroup_FPR+"/run_19FP_InsSex.csv")
  FP4_InsSex = pd.read_csv(twogroup_FPR+"/run_31FP_InsSex.csv")
  FP3_InsSex = pd.read_csv(twogroup_FPR+"/run_38FP_InsSex.csv")
  FP2_InsSex = pd.read_csv(twogroup_FPR+"/run_47FP_InsSex.csv")
  FP1_InsSex = pd.read_csv(twogroup_FPR+"/run_77FP_InsSex.csv")

  FP_InsSex  =pd.concat([FP1_InsSex,FP2_InsSex, FP3_InsSex,FP4_InsSex, FP5_InsSex])
  FP_InSx =FP_InsSex.groupby("Insurance")
  FP_InSx_df = FP_InSx.describe()
  
  factors = ['FPR_F', 'FPR_M']
  Insurance = ['Medicaid','Medicare','Other']
  SexIns_df = pd.DataFrame(Insurance, columns=["Insurance"])

  InsSex_df = FiveRun(factors,SexIns_df,FP_InSx_df,significance_level)
  InsSex_df.to_csv(twogroup_FPR+'/Inter_SexIns.csv',index=False)

def get_avg_InsRace_fpr(path,significance_level=1.96):
  
  twogroup_FPR=path
  
  FP5_InsRace = pd.read_csv(twogroup_FPR+"/run_19FP_InsRace.csv")
  FP4_InsRace = pd.read_csv(twogroup_FPR+"/run_31FP_InsRace.csv")
  FP3_InsRace = pd.read_csv(twogroup_FPR+"/run_38FP_InsRace.csv")
  FP2_InsRace = pd.read_csv(twogroup_FPR+"/run_47FP_InsRace.csv")
  FP1_InsRace = pd.read_csv(twogroup_FPR+"/run_77FP_InsRace.csv")

  FP_InsRace  =pd.concat([FP1_InsRace,FP2_InsRace, FP3_InsRace,FP4_InsRace, FP5_InsRace])
  FP_InsRace =FP_InsRace.groupby("race")
  FP_InRa_df =FP_InsRace.describe()

  factors = ['FPR_Medicare', 'FPR_Other','FPR_Medicaid']
  race = ['AMERICAN INDIAN/ALASKA NATIVE','ASIAN','BLACK/AFRICAN AMERICAN','HISPANIC/LATINO','OTHER','WHITE']
  InsRace_df = pd.DataFrame(race, columns=["race"])

  InsRace_df = FiveRun(factors,InsRace_df,FP_InRa_df,significance_level)
  InsRace_df.to_csv(twogroup_FPR+'/Inter_RaceIns.csv',index=False)


def get_avg_InsAge_fpr(path,significance_level=1.96):
  
  twogroup_FPR=path
  
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FP_InsAge.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  FP5_InsAge = pd.read_csv(twogroup_FPR+"/run_19FP_InsAge.csv")
  FP4_InsAge = pd.read_csv(twogroup_FPR+"/run_31FP_InsAge.csv")
  FP3_InsAge = pd.read_csv(twogroup_FPR+"/run_38FP_InsAge.csv")
  FP2_InsAge = pd.read_csv(twogroup_FPR+"/run_47FP_InsAge.csv")
  FP1_InsAge = pd.read_csv(twogroup_FPR+"/run_77FP_InsAge.csv")

  FP_InsAge =pd.concat([FP1_InsAge,FP2_InsAge, FP3_InsAge,FP4_InsAge, FP5_InsAge])

  FP_InsAge =FP_InsAge.groupby("age")
  FP_InsAge_df =FP_InsAge.describe()

  factors = ['FPR_Medicare', 'FPR_Other','FPR_Medicaid']
  age = ['0-20','20-40','40-60','60-80','80-']
  InsAge_df = pd.DataFrame(age, columns=["age"])

  InsAge_df = FiveRun(factors,InsAge_df,FP_InsAge_df,significance_level)
  InsAge_df.to_csv(twogroup_FPR+'/Inter_AgeIns.csv',index=False)


def get_avg_RaceAgeSex_fpr(path,significance_level=1.96):
  
  #three_group_dir = './FPR/Three_Group_Intersection_FPR'
  three_group_dir = path
  
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FP_RaceAgeSex.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)

  RaceAgeSex_19 = pd.read_csv(three_group_dir+"/run_19FP_RaceAgeSex.csv")
  RaceAgeSex_31 = pd.read_csv(three_group_dir+"/run_31FP_RaceAgeSex.csv")
  RaceAgeSex_38 = pd.read_csv(three_group_dir+"/run_38FP_RaceAgeSex.csv")
  RaceAgeSex_47 = pd.read_csv(three_group_dir+"/run_47FP_RaceAgeSex.csv")
  RaceAgeSex_77 = pd.read_csv(three_group_dir+"/run_77FP_RaceAgeSex.csv")


  fp_race_age_sex_df =pd.concat([RaceAgeSex_19, RaceAgeSex_31,RaceAgeSex_38,
                        RaceAgeSex_47,RaceAgeSex_77])

  fp_race_age_sex_df =fp_race_age_sex_df.groupby("race")
  fp_race_age_sex_df = fp_race_age_sex_df.describe()


  factors = ['FPR_60_80_M', 'FPR_40_60_M','FPR_20_40_M','FPR_80+_M','FPR_0_20_M','FPR_60_80_F',
            'FPR_40_60_F','FPR_20_40_F','FPR_80+_F','FPR_0_20_F']
  race = ['AMERICAN INDIAN/ALASKA NATIVE','ASIAN','BLACK/AFRICAN AMERICAN','HISPANIC/LATINO','OTHER','WHITE']
  RaceAgeSex_df = pd.DataFrame(race, columns=["race"])

  RaceAgeSex_df = FiveRun(factors,RaceAgeSex_df,fp_race_age_sex_df,significance_level)

  RaceAgeSex_df.to_csv(three_group_dir+'/Inter_RaceAgeSex.csv')


def get_avg_RaceInsSex_fpr(path,significance_level=1.96):
  
  #three_group_dir = './FPR/Three_Group_Intersection_FPR'
  three_group_dir = path
  
  # Read the CSV files for each seed
  # The files are expected to be in the format: run_<seed>FP_RaceInsSex.csv
  # where <seed> is the seed number (19, 31, 38, 47, 77)
  
  RaceInsuSex_19 = pd.read_csv(three_group_dir+"/run_19FP_RaceInsSex.csv")
  RaceInsuSex_31 = pd.read_csv(three_group_dir+"/run_31FP_RaceInsSex.csv")
  RaceInsuSex_38 = pd.read_csv(three_group_dir+"/run_38FP_RaceInsSex.csv")
  RaceInsuSex_47 = pd.read_csv(three_group_dir+"/run_47FP_RaceInsSex.csv")
  RaceInsuSex_77 = pd.read_csv(three_group_dir+"/run_77FP_RaceInsSex.csv")

  fp_race_insu_sex_df =pd.concat([RaceInsuSex_19, RaceInsuSex_31,RaceInsuSex_38,
                        RaceInsuSex_47,RaceInsuSex_77])

  fp_race_insu_sex_df =fp_race_insu_sex_df.groupby("race")
  fp_race_insu_sex_df = fp_race_insu_sex_df.describe()


  factors = ['FPR_M_Medicare', 'FPR_M_Other','FPR_M_Medicaid','FPR_F_Medicare','FPR_F_other','FPR_F_Medicaid']
  race = ['WHITE','BLACK/AFRICAN AMERICAN','HISPANIC/LATINO','OTHER','ASIAN','AMERICAN INDIAN/ALASKA NATIVE']
  RaceInsuSex_df = pd.DataFrame(race, columns=["race"])

  RaceInsuSex_df = FiveRun(factors,RaceInsuSex_df,fp_race_insu_sex_df,significance_level)

  RaceInsuSex_df.to_csv(three_group_dir+'/Inter_RaceInsuSex.csv')



def main():
  
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
  get_avg_subgroup_insurance_fpr(subgroup_FPR,significance_level)
  
  
  # Two group average FPR
  
  twogroup_FPR="./FPR/Two_Group_Intersection_FPR"
  get_avg_AgeSex_fpr(twogroup_FPR,significance_level)
  get_avg_RaceSex_fpr(twogroup_FPR,significance_level)
  get_avg_RaceAge_fpr(twogroup_FPR,significance_level)
  get_avg_InsSex_fpr(twogroup_FPR,significance_level)
  get_avg_InsRace_fpr(twogroup_FPR,significance_level)
  get_avg_InsAge_fpr(twogroup_FPR,significance_level)
  
  # Three group average FPR
  three_group_dir = './FPR/Three_Group_Intersection_FPR'
  get_avg_RaceAgeSex_fpr(three_group_dir,significance_level)
  get_avg_RaceInsSex_fpr(three_group_dir,significance_level)
  

if __name__ == "__main__":
  warnings.filterwarnings("ignore")
  main()
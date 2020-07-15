import numpy as np
import pandas as pd
import math
import os


class Categorical_Encoding():
  
    param_dict_Col = {}
    def __init__ (self, strategy, variables,Topn=None, Target_Var= None):
        self.strategy = strategy
        self.variables = variables
        self.Target_Var = Target_Var
        self.Topn = Topn
        
    def fit(self,df):        
        if self.strategy =='OneHot':
            for i in self.variables:
                varList = df[i].unique()
                self.param_dict_Col[i]= varList[:self.Topn] 
                
        if self.strategy == 'Ratio_Encoding':
            for i in self.variables:                
                df[i].fillna('Missing',inplace=True)
                df['Prob_Target_1'] = df[self.Target_Var]
                prob_df = pd.DataFrame(df.groupby([i])['Prob_Target_1'].mean())                
                prob_df['Prob_Target_0'] =  1- prob_df.Prob_Target_1
                prob_df['Ratio'] = prob_df.Prob_Target_1/prob_df.Prob_Target_0                 
                self.param_dict_Col[i]= prob_df['Ratio'].to_dict()   
                
        if self.strategy == 'WOE_Encoding':
            for i in self.variables:
                df[i].fillna('Missing',inplace=True)
                per_df = pd.DataFrame(pd.crosstab(df[i], df[self.Target_Var], normalize='columns').mul(100))
                per_df.rename(columns={0: "Target_0_Per", 1: "Target_1_Per"},inplace=True)
                per_df['WOE'] = np.log(per_df['Target_1_Per']/per_df['Target_0_Per'])
                value = per_df['WOE'].to_dict()        
                self.param_dict_Col[i]= value
                
        return self
                    
            
            
    def transform(self,df):
        if self.strategy =='OneHot':
            for key,value in self.param_dict_Col.items():
                for a in value:
                    column = key + "_" + a
                    df[column] = np.where(df[key] == a,1,0)  
                    
        if self.strategy == 'Ratio_Encoding':
            for var in self.param_dict_Col:
                df[str(var) +"_Encoded"] = df[var].map(self.param_dict_Col[var] )      
                
        if self.strategy == 'WOE_Encoding':
            for var in self.param_dict_Col:
                df[str(var) +"_WOE_Encoded"] = df[var].map(self.param_dict_[var])
        
        return df       
        

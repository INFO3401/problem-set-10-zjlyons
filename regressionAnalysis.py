#Zach Lyons
#INFO 3401
#Problem Set 9
#Worked with Harold, Steve, Justin, and Luke. 

#Imports
import pandas as pd
import csv
import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Part A
class AnalysisData:
    def __init__(self):
        self.dataset = []
        self.variables = []
    
    def parseFile(self, filename):
        self.dataset = pd.read_csv(filename)
        for column in self.dataset.columns.values:
            if column != "competitorname":
                self.variables.append(column)

                
#Part B
class LinearAnalysis:
    def __init__(self, target_Y):
        self.targetY = target_Y
        self.bestX = ""
    
    def runSimpleAnalysis(self, data):
        best_rscore = -1
        best_variable = ""
        
        for column in data.variables:
                if column != self.targetY:
                        idp_var = data.dataset[column].values
                        idp_var = idp_var.reshape(len(idp_var), 1)
                        
        regr = LinearRegression()
        regr.fit(idp_var, data.dataset[self.targetY])
       
        prediction = regr.predict(idp_var)
        
        r_score = r2_score(data.dataset[self.targetY], prediction)
        
        if r_score > best_rscore:
            best_rscore = r_score
            best_variable = column
            
        self.bestX = best_variable
        print(best_variable, best_rscore)
        
#Part C, to be continued...
class LogisticAnalysis:
    def __init__(self, target_Y):
        self.targetY = target_Y
        self.bestX = ""
        

#Questions 1, 2, and 3
Analysis_Data = AnalysisData()
Analysis_Data.parseFile('candy-data.csv')

Linear_Analysis = LinearAnalysis('sugarpercent')
Linear_Analysis.runSimpleAnalysis(Analysis_Data)
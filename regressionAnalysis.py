#Zach Lyons
#INFO 3401
#Problem Set 9/10 - regressionAnalysis.py
#Worked with Harold, Steve, Justin, and Luke. 

#Note: I fixed some indentation errors from problem set 9 that were producing incorect "Best Variables".     

#Imports
import pandas as pd
import csv
import matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
        print('Best Variable: ' + best_variable, best_rscore)
        print ('Coefficient: ' + str(regr.coef_))
        print ('Intercept: ' + str(regr.intercept_))
        
#Part C
class LogisticAnalysis:
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
                        
                        regr = LogisticRegression()
                        regr.fit(idp_var, data.dataset[self.targetY])
       
                        prediction = regr.predict(idp_var)
        
                        r_score = r2_score(data.dataset[self.targetY], prediction)
        
                        if r_score > best_rscore:
                            best_rscore = r_score
                            best_variable = column
            
        self.bestX = best_variable
        print('Best Variable: ' + best_variable, best_rscore)
        print ('Coefficient: ' + str(regr.coef_))
        print ('Intercept: ' + str(regr.intercept_))
        
    def runMultipleRegression(self, data):
        multi_regr = LogisticRegression()
        independent_vars = [val for val in data.variables if val != self.targetY]
        multi_regr.fit(data.dataset[independent_vars], data.dataset[self.targetY])
        prediction = multi_regr.predict(data.dataset[independent_vars])
        r_score = r2_score(data.dataset[self.targetY], prediction)
        
        print ('Coefficients: ' + str(multi_regr.coef_))
        print ("Intercept: " + str(multi_regr.intercept_))
        print('r Squared: ' + str(r_score))
        

#Testing Code

Analysis_Data = AnalysisData()
Analysis_Data.parseFile('candy-data.csv')

print("Linear Regression Analysis")
Linear_Analysis = LinearAnalysis('sugarpercent')
Linear_Analysis.runSimpleAnalysis(Analysis_Data)

print("Logistic Regression Analysis")
Logistic_Analysis = LogisticAnalysis('chocolate')
Logistic_Analysis.runSimpleAnalysis(Analysis_Data)

print("Multiple Regression Analysis")
Logistic_Analysis.runMultipleRegression(Analysis_Data)
print(Analysis_Data.variables)

#Regression Formulas (based on script outputs)

#Linear : predicted sugarpercent = 0.257 + (0.044 * pricepercent)

#Logistic: predicted sugarpercent = -3.088 + (0.059 * fruity)

#Multiple: predicted sugarpercent = -1.682 + (-2.529 * chocolate) + (-0.197 * fruity) + (0.039 * caramel) + (-0.165 * peanutyalmondy) + (0.498 * nougat) + (-0.476 * crispedricewafer) + (0.815 * hard) + (-0.6 * bar) + (-0.258 * pluribus) + (0.322 * pricepercent) + (0.054 * winpercent)

#Friday Problems - Problem Set 10

#Problem a
#independent: chocolate and caramel - categorical
#dependent: sugar percentage - continuous
#null: chocolate and caramel candy have the same amount of sugar

#Problem b
#independent: blue state/red state - categorical
#dependent: split tickets - continuous
#null:blue and red states have the same amount of split ticket voters

#Problem c
#independent: battery life - continuous
#dependent: rate of sales - continuous
#null: battery life does not affect sales of cell phones
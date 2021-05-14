import pandas as pd
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
import itertools as itr
import math
import os
from sklearn import preprocessing

'''
#load data to be used for test, as pandas dataframe; class_label is the name of class column
def load_data(filename, class_label):
	
	if filename[-4:]=='.csv':
		data=pd.read_csv(filename)
		y_data=data[class_label]
		data.drop([class_label], axis=1,inplace=True)
	else:
		if filename[-5:]=='.xlsx':
			data=pd.read_excel(filename)
			y_data=data[class_label]
			data.drop([class_label], axis=1,inplace=True)
		else:
			if filename[-4:]=='.txt':
				data=pd.read_csv(filename,delimiter="\t")
				y_data=data[class_label]
				data.drop([class_label], axis=1,inplace=True)
	print(y_data)
	# if ouput values are not in {0,1} (e.g. categorical), convert it			
	if not y_data.isin([0,1]).all():
		le = preprocessing.LabelEncoder()
		le.fit(y_data)
		y_data=le.transform(y_data)
	print(y_data)
	return (data,y_data)
'''


#compute min and max acceptable deltas to find new thresholds within feature ranges;
# Nf: # of perturbed features
# ths: list of Nf old threshold values
# fmin: list of Nf minimum values of features
#fmax: list of Nf maximum values for features
#cond_format: list of Nf values: if 0, perturbation is th-abs(th*delta); if 1 perturbation is th+abs(th*delta)
def get_delta_limits(Nf,ths,fmin,fmax, cond_format):
	delta=Symbol('delta',real=True)
	delta_limits=np.zeros((Nf,2))
	for i in range(Nf):
		if cond_format[i]==0:
			sol=str(solve([(ths[i]-abs(ths[i]*delta))>=fmin[i], (ths[i]-abs(ths[i]*delta))<=fmax[i]]))
			delta_min=float(sol.split('&')[0][1:][:-10])
			delta_max=float(sol.split('&')[1][10:][:-1])
			delta_limits[i]=np.asarray((delta_min,delta_max))
		else:
			if cond_format[i]==1:
				sol=str(solve([(ths[i]+abs(ths[i]*delta))>=fmin[i], (ths[i]+abs(ths[i]*delta))<=fmax[i]]))
				delta_min=float(sol.split('&')[0][1:][:-10])
				delta_max=float(sol.split('&')[1][10:][:-1])
				delta_limits[i]=np.asarray((delta_min,delta_max))
	return delta_limits

# get deltas acceptable ranges, index is the feature: 0-->1st feature, 1-->2nd feat etc
def get_delta_range(delta_limits,step):
	delta_ranges=[]
	for i in range(len(delta_limits)):
		delta_ranges.append(np.arange(delta_limits[i][0],delta_limits[i][1],step[i]))
	return delta_ranges

#per la coppia delta1, delta2 trova il vettore predicted con le predizioni date dalle regole considerate
#data: dataset di test
#delta1, delta2 perturbazioni delle soglie
#ths: lista soglie originali
#flabels: label delle features da perturbare
#method: 'inside' o 'outside'
#case: per forme degli intervalli--> 1 (f1>th1, f2<th2), 2 (f1<th1, f2>th2); 3 (f1>th1, f2>th2); 4 (f1<th1, f2<th2)
def get_prediction(data,delta1,delta2,ths,flabels,method,case):
	th1=ths[0]
	th2=ths[1]
	f1=flabels[0]
	f2=flabels[1]
	predicted=[]
	if method=='outside':
		if case==1:
			for i in range(0,len(data)):
				if (data[f1][i]>th1-abs(th1*delta1)  or data[f2][i]<th2+abs(th2*delta2)):
					predicted.append(1)
				else:
					predicted.append(0)
			return predicted
		else:
			if case==2:
				for i in range(0,len(data)):
					if (data[f1][i]<th1+abs(th1*delta1)  or data[f2][i]>th2-abs(th2*delta2)):
						predicted.append(1)
					else:
						predicted.append(0)
				return predicted
			else:
				if case==3:
					for i in range(0,len(data)):
						if (data[f1][i]>th1-abs(th1*delta1)  or data[f2][i]>th2-abs(th2*delta2)):
							predicted.append(1)
						else:
							predicted.append(0)
					return predicted
				else:
					if case==4:
						for i in range(0,len(data)):
							if (data[f1][i]<th1+abs(th1*delta1)  or data[f2][i]<th2+abs(th2*delta2)):
								predicted.append(1)
							else:
								predicted.append(0)
						return predicted
	else:
		if method=='inside':
			if case==1:
				for i in range(0,len(data)):
					if (data[f1][i]>th1+abs(th1*delta1)  or data[f2][i]<th2-abs(th2*delta2)):
						predicted.append(0)
					else:
						predicted.append(1)
				return predicted
			else:
				if case==2:
					for i in range(0,len(data)):
						if (data[f1][i]<th1-abs(th1*delta1)  or data[f2][i]>th2+abs(th2*delta2)):
							predicted.append(0)
						else:
							predicted.append(1)
					return predicted
				else:
					if case==3:
						for i in range(0,len(data)):
							if (data[f1][i]>th1+abs(th1*delta1)  or data[f2][i]>th2+abs(th2*delta2)):
								predicted.append(0)
							else:
								predicted.append(1)
						return predicted
					else:
						if case==4:
							for i in range(0,len(data)):
								if (data[f1][i]<th1-abs(th1*delta1)  or data[f2][i]<th2-abs(th2*delta2)):
									predicted.append(0)
								else:
									predicted.append(1)
							return predicted



def get_eval_metrics(predicted,y_data):
	TP=0
	FP=0
	TN=0
	FN=0
	for i in range(0,len(y_data)):
		if (predicted[i]==1 and y_data[i]==1):
		    TP+=1
		else: 
		    if (predicted[i]==1 and y_data[i]==0):
		        FP+=1
		    else: 
		        if (predicted[i]==0 and y_data[i]==0):
		            TN+=1
		        else:
		            if (predicted[i]==0 and y_data[i]==1):
		                FN+=1
	return (TP,FP,TN,FN)

def rule_perturbation(data, y_data,ths,flabels, delta_limits, output_filename,method, case):
	oom1=math.floor(math.log10(delta_limits[0][1]))
	step1=10**oom1/2
	#step1=0.05
	oom2=math.floor(math.log10(delta_limits[1][1]))
	step2=10**oom2/2
	print(step1)
	print(step2)
	delta_ranges=get_delta_range(delta_limits,[step1,step2])
	if os.path.isfile(output_filename):
		os.remove(output_filename)
	with open(output_filename,'a') as output:
		output.write("Delta1"+","+"Delta2"+","+"Error"+","+"Coverage"+","+"TP, FP, TN, FN"+"\n")
	for delta1,delta2 in itr.product(delta_ranges[0],delta_ranges[1]):
		predicted=get_prediction(data,delta1,delta2,ths,flabels,method,case)
		#confronto di predicted con gli output veri per trovare il totale dei TP, FP,FN,TN
		TP,FP,TN,FN=get_eval_metrics(predicted,y_data)
		error=FN/(FN+TP)
		coverage=TN/(TN+FP)
		with open(output_filename,'a') as output:
			output.write(str(delta1)+","+str(delta2)+","+str(error)+","+str(coverage)+','+str(TP)+','+str(FP)+','+str(TN)+','+str(FN)+'\n')









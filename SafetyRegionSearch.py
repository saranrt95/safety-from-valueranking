import itertools as itr
import numpy as np
from getDeltaMatrix import *
import pandas as pd
import operator as op
from config import *
#from safetyregionEval import *

operators = {'<': op.lt,
			'<=': op.le,
			'=': op.eq,
			'>': op.gt,
			'>=': op.ge}

def VerifyLogicalOr(datarow, flabels, Nf, vrops, thops, deltarange, method):
	r = 0
	for i in range(Nf):
		
		if vrops[i].find('<')!=-1 and method == "outside":
			if operators[vrops[i]](datarow[flabels[i]],thops[i]+abs(deltarange[i]*thops[i])):
				r+=1
		if vrops[i].find('>')!=-1 and method == "outside":
			if operators[vrops[i]](datarow[flabels[i]],thops[i]-abs(deltarange[i]*thops[i])):
				r+=1
		if vrops[i].find('<')!=-1 and method == "inside":
			if operators[vrops[i]](datarow[flabels[i]],thops[i]-abs(deltarange[i]*thops[i])):
				r+=1
		if vrops[i].find('>')!=-1 and method == "inside":
			if operators[vrops[i]](datarow[flabels[i]],thops[i]+abs(deltarange[i]*thops[i])):
				r+=1
		
	return r

def getNewThresholds(intervals, optimalDelta, method):
	#print(optimalDelta)
	Nf = len(intervals)
	thops = getOriginalThresholds(intervals)
	vrops = getOperators(intervals)
	newths =[]
	for i in range(Nf):
		if vrops[i].find('<')!=-1 and method == "outside":
			newths.append(thops[i]+abs(optimalDelta[i]*thops[i]))
			
		if vrops[i].find('>')!=-1 and method == "outside":
			newths.append(thops[i]-abs(optimalDelta[i]*thops[i]))

		if vrops[i].find('<')!=-1 and method == "inside":
			newths.append(thops[i]-abs(optimalDelta[i]*thops[i]))

		if vrops[i].find('>')!=-1 and method == "inside":
			newths.append(thops[i]+abs(optimalDelta[i]*thops[i]))
	return newths


def print_safety_region(method, flabels, vrops, new_ths):
	finalsentence=""
	if method=="outside":
		# operators for complementary
		opmapping = {'>':'<','<':'>','>=':'<=','<=':'>='}
		for i in range(len(flabels)):
			if i!=len(flabels)-1:
				finalsentence = finalsentence+flabels[i]+" "+opmapping[vrops[i]]+" "+str(new_ths[i])+" AND "
			else:
				finalsentence = finalsentence + flabels[i]+" "+opmapping[vrops[i]]+" "+str(new_ths[i])
	if method == "inside":
		for i in range(len(flabels)):
			if i!=len(flabels)-1:
				finalsentence = finalsentence+ flabels[i]+" "+vrops[i]+" "+str(new_ths[i])+" OR "
			else:
				finalsentence = finalsentence + flabels[i]+" "+vrops[i]+" "+str(new_ths[i])
	#print("finalsentence: ", finalsentence)
	return finalsentence


# main algorithms
def getSafetyRegion(data, y_data, d1, flabels, intervals, delta_ranges, method, save_res=False, minimizeFPR = False):
	#print("d1: ",d1)
	# intervals: lista di tuple di lunghezza 3 (flabel, operator, threshold) o di lunghezza 5 (non ancora implementato)
	Nf = len(intervals)
	vrops = getOperators(intervals)
	thops = getOriginalThresholds(intervals)
	deltalist=[]
	FPRlist=[]
	TPRlist=[]
	FNRlist=[]
	TNRlist=[] 
	acclist =[]
	F1list =[]
	PPVlist = []
	NPVlist = []
	if save_res:
		with open("output_"+str(method)+".csv","a") as resfile:
				resfile.write("Delta Vector,FNR,TNR,FPR,TPR,Accuracy,F1score\n")
	for [*deltarange] in itr.product(*delta_ranges):
		#print("delta range: ",deltarange)
		# delta range lista di Nf elementi
		TP = 0 
		TN = 0 
		FP = 0
		FN = 0

		for j in range(len(data)):
			r = VerifyLogicalOr(data.iloc[j,:], flabels, Nf, vrops, thops, deltarange, method)
			# verified for r!=0
			if minimizeFPR:
				if r!=0:
					if method == "outside":
						predicted = 0
					if method == "inside":
						predicted = 1
				else:
					if method == "outside":
						predicted = 1
					if method == "inside":
						predicted = 0			
			else:

				if r!=0:
					if method == "outside":
						predicted = 1
					if method == "inside":
						predicted = 0
				else:
					if method == "outside":
						predicted = 0
					if method == "inside":
						predicted = 1
			if predicted == 1 and y_data[j]==1: TP+=1
			elif predicted == 0 and y_data[j]==0: TN+=1
			elif predicted == 1 and y_data[j]==0: FP+=1
			elif predicted == 0 and y_data[j]==1: FN+=1

		FNR=FN/(FN+TP) # false negative rate
		TNR=TN/(TN+FP)#true negative rate
		FPR=FP/(FP+TN) #false positive rate
		TPR=TP/(TP+FN) #true positive rate
		Acc=(TP+TN)/(TP+TN+FP+FN)
		f1score=(2*TP)/(2*TP+FP+FN)
		if TP+FP!=0:
			PPV = TP/(TP+FP)
		else:
			PPV=0
		if TN+FN!=0:
			NPV = TN/(TN+FN)
		else:
			NPV = 0
		FNRlist.append(FNR)
		FPRlist.append(FPR)
		TPRlist.append(TPR)
		TNRlist.append(TNR)
		acclist.append(Acc)
		F1list.append(f1score)
		PPVlist.append(PPV)
		NPVlist.append(NPV)
		if save_res:
			with open("output_"+str(method)+".csv","a") as resfile:
				resfile.write(str(deltarange)+","+str(FNR)+","+str(TNR)+","+str(FPR)+","+str(TPR)+","+str(Acc)+","+str(f1score)+","+str(PPV)+","+str(NPV)+"\n")

		deltalist.append(deltarange)

	outputRes=pd.DataFrame(list(zip(deltalist,FPRlist,TPRlist,TNRlist,FNRlist,acclist,F1list,PPVlist,NPVlist)), columns = ["Delta-Vector", "FPR","TPR", "TNR","FNR","Accuracy","F1-score","PPV","NPV"])
	if minimizeFPR:
		minErr=min(outputRes['FPR'])
		outErrmin=outputRes.loc[outputRes['FPR']==minErr]
		maxCov=max(outErrmin['TPR'])
		outMaxCov=outErrmin.loc[outErrmin['TPR']==maxCov]
	else:
		minErr=min(outputRes['FNR'])
		outErrmin=outputRes.loc[outputRes['FNR']==minErr]
		maxCov=max(outErrmin['TNR'])
		outMaxCov=outErrmin.loc[outErrmin['TNR']==maxCov]
	# TODO: filtrare sui delta
	# TODO: vedere di restituire direttamente la regione come intervallo e le metriche associate
	#print(outMaxCov["Delta-Vector"])
	#print("New thresholds: ", getNewThresholds(intervals, outMaxCov["Delta-Vector"].values[0], method))
	new_ths =  getNewThresholds(intervals, outMaxCov["Delta-Vector"].values[0], method)
	safety_reg = print_safety_region(method, flabels, vrops, new_ths)
	return safety_reg, outMaxCov["Delta-Vector"].values[0], outMaxCov[["FNR","TNR", "TPR","FPR","Accuracy","F1-score","PPV","NPV"]].values[0]




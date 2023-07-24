import sys
import pandas as pd
import time
from parseVR import *
from load_dataset import *
from getDeltaMatrix import *
from config import *
from SafetyRegionSearch import *


def evaluateSolution(data, y_data, intervals, optimalDelta, method, minimizeFPR=False):

	th_orig = getOriginalThresholds(intervals)
	Nf = len(intervals)
	vrops = getOperators(intervals)
	flabels = getFeatureLabels(intervals)
	TP=0
	FP=0
	FN=0
	TN=0
	for j in range(len(data)):
		r = VerifyLogicalOr(data.iloc[j,:], flabels, Nf, vrops, th_orig, optimalDelta, method)
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
		if predicted == 0 and y_data[j]==0: TN+=1
		if predicted == 1 and y_data[j]==0: FP+=1
		if predicted == 0 and y_data[j]==1: FN+=1

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
	#newths = getNewThresholds(intervals, optimalDelta, method)
	#print("new thresholds: ",newths)
	return FNR,TNR,TPR,FPR,Acc,f1score,PPV,NPV


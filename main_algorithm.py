import sys
import pandas as pd
import time

from config import *

from parseVR import *
from load_dataset import *
from getDeltaMatrix import *
#from safetyregionssearch import *

from SafetyRegionSearch import *
from safetyregionEval import *

def SafetyFromValueRanking(vrstringlist,method,datafilename, class_label, steps,testfile=None, save_res = False, minimizeFPR = False):
	#print(minimizeFPR)
	#print(vrstringlist)
	initialregion = print_initial_region(vrstringlist)
	print("INITIAL REGION: \n")
	print(initialregion+"\n")

	# get elements from value ranking
	intervals=getValueRankingInfo(vrstringlist)

	# get feature labels of interest
	flabels = getFeatureLabels(intervals)

	# load data and extract only the features of interest
	train_data, y_train = load_data(datafilename, class_label, flabels)
	# mod: if minimizeFPR, exchange the 0s and 1s
	# number of points from the unsafe class
	D1 = len(y_train[y_train==1])
	#print(type(steps[0]))
	# get the candidate perturbations for each feature
	delta_ranges = getPerturbationLimits(train_data,y_train,intervals, method,steps)
	#print(delta_ranges)
	print("Training with {} method\n".format(method))
	t_start = time.time()
	safetyReg, optimalDelta, metrics = getSafetyRegion(train_data, y_train, D1, flabels, intervals, delta_ranges, method,save_res,minimizeFPR)
	t_end = time.time()
	print("SAFETY REGION: \n")
	print(safetyReg)
	print("FNR = {:.2f}, TNR = {:.2f}, TPR = {:.2f}, FPR = {:.2f}, Accuracy = {:.2f}, F1score = {:.2f}, PPV = {:.2f}, NPV = {:.2f}\n".format(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5],metrics[6],metrics[7]))
	print("elapsed time for training: ", t_end - t_start, "s\n")


	if testfile!=None:
		print("*"*50)
		print("Evaluating the region on the test set\n")
		test_data, y_test = load_data(testfile, class_label, flabels)

		FNR, TNR, TPR, FPR, accuracy, f1,PPV,NPV = evaluateSolution(test_data, y_test, intervals, optimalDelta, method, minimizeFPR)
		print("FNR = {:.2f}, TNR = {:.2f}, TPR = {:.2f}, FPR = {:.2f}, Accuracy = {:.2f}, F1score = {:.2f}, PPV = {:.2f}, NPV = {:.2f}\n".format(FNR, TNR, TPR, FPR, accuracy, f1, PPV, NPV))

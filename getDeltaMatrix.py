from sympy.solvers import solve
from sympy import Symbol
from sympy.solvers.inequalities import reduce_rational_inequalities

from config import *

import itertools as itr
import math
import os
import numpy as np

from parseVR import *

import operator as op

operators = {'<': op.lt,
			'<=': op.le,
			'=': op.eq,
			'>': op.gt,
			'>=': op.ge}


def getMinMaxValuesForFeatures(data):

	fmaxvalues = data.max(axis=0)
	fminvalues = data.min(axis=0)
	return fminvalues, fmaxvalues



def getDeltaSteps(delta_limits):
	# steps are taken based on the order of magnitude 
	steps=[]
	for i in range(delta_limits.shape[0]):
		#print(delta_limits[i][1])
		oom_i=math.floor(math.log10(delta_limits[i][1]))
		#print(oom_i)
		step_i=10**oom_i/2
		#step_i = 1
		print("step_i: ",step_i)
		steps.append(step_i)
	return steps

def parse_solution(sol):
	delta_min=float(sol.split('&')[0][1:][:-10])
	delta_max=float(sol.split('&')[1][10:][:-1])
	deltas=np.asarray((delta_min,delta_max))
	return deltas
'''
def VerifyStartingOr(datarow, flabels, Nf, vrops, thops):
	r = 0
	for i in range(Nf):
		
		if vrops[i].find('<')!=-1:
			if operators[vrops[i]](datarow[flabels[i]],thops[i]):
				r+=1
		
	return r

def getTuningDirection(method,data,y,intervals):

	Nf = len(intervals)
	vrops = getOperators(intervals)
	thops = getOriginalThresholds(intervals)
	flabels = getFeatureLabels(intervals)
	n = 0 # number of starting class points inside the proposed region
	for r in range(len(data)):
		X_row = data.iloc[r,:]
		# check if the point X_row satisfies the OR of value ranking intervals
		verified = VerifyStartingOr(X_row, flabels, Nf, vrops, thops)
		if not minimizeFPR:
			if method == "outside":
				if verified!=0 and y[r]==1:
					n+=1
			else:
				if method == "inside":
					if verified!=0 and y[r]==0:
						n+=1
		else:
			if method == "outside":
				if verified!=0 and y[r]==0:
					n+=1
			else:
				if method == "inside":
					if verified!=0 and y[r]==1:
						n+=1
	# get the number of points of the starting class outisde the proposed region
	if not minimizeFPR:
		if method == "outside":
			m = len(y[y==1]) 
		else:
			if method == "inside":
				m = len(y[y==0])
	else:
		if method == "outside":
			m = len(y[y==0])
		else:
			if method == "inside":
				m = len(y[y==1])
	return n > m-n		

'''
def getPerturbationLimits(data,y,intervals, method,steps):

	Nf = len(intervals)
	
	delta=Symbol('delta',real=True)
	delta_limits=np.zeros((Nf,2))
	th = None
	th1 = None
	th2 = None

	
	fmin,fmax = getMinMaxValuesForFeatures(data)
	fminlist = list(fmin)
	fmaxlist = list(fmax)
	thresholds = getOriginalThresholds(intervals)
	operatorslist = getOperators(intervals)
	# NEW wrt to publications
	#tuning_direction = getTuningDirection(method,data,y,intervals)
	
	tuning_direction = True
	for i in range(Nf):
		#print("i = ",i)
		if type(thresholds[i]) is tuple and len(thresholds[i])==2 and len(operatorslist[i])==2:
			th1 = thresholds[i][0]
			op1 = operatorslist[i][0]
			th2 = thresholds[i][1]
			op2 = operatorslist[i][1] 
		else:
			th = thresholds[i]
			opp = operatorslist[i]
		
		# single thresholds
		if th!=None and opp.find('<')!=-1:
			if method == "outside":
				if tuning_direction:
					sol=str(solve([(th+abs(th*delta))>=fminlist[i], (th+abs(th*delta))<=fmaxlist[i]]))
				else:
					sol=str(solve([(th-abs(th*delta))>=fminlist[i], (th-abs(th*delta))<=fmaxlist[i]]))
				delta_limits[i] =parse_solution(sol)
			else:
				if method == "inside":
					if tuning_direction:
						sol=str(solve([(th-abs(th*delta))>=fminlist[i], (th-abs(th*delta))<=fmaxlist[i]]))
					else:
						sol=str(solve([(th+abs(th*delta))>=fminlist[i], (th+abs(th*delta))<=fmaxlist[i]]))
					delta_limits[i] =parse_solution(sol)
		
		if th!=None and opp.find('>')!=-1:
			if method == "outside":
				if tuning_direction:
					sol=str(solve([(th-abs(th*delta))>=fminlist[i], (th-abs(th*delta))<=fmaxlist[i]]))
				else:
					sol=str(solve([(th+abs(th*delta))>=fminlist[i], (th+abs(th*delta))<=fmaxlist[i]]))
				delta_limits[i] =parse_solution(sol)
			else:
				if method == "inside":
					if tuning_direction:
						sol=str(solve([(th+abs(th*delta))>=fminlist[i], (th+abs(th*delta))<=fmaxlist[i]]))
					else:
						sol=str(solve([(th-abs(th*delta))>=fminlist[i], (th-abs(th*delta))<=fmaxlist[i]]))
					delta_limits[i] =parse_solution(sol)
				
		# DOPPIA SOGLIA (TODO NEXT)
		if th1!=None and th2!=None:
			if method == "outside":
				pass
			else:
				if method == "inside":
					pass

	delta_ranges=[]
	for i in range(delta_limits.shape[0]):
		delta_ranges.append(np.arange(delta_limits[i][0],delta_limits[i][1]+steps[i],steps[i]))

	return delta_ranges




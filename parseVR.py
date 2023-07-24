def print_initial_region(vrstringlist):
	initialregion=""
	for i in range(len(vrstringlist)):
		if i!=len(vrstringlist)-1:
			initialregion+=vrstringlist[i]+" OR "
		else:
			initialregion+=vrstringlist[i]
	return initialregion

def ParseVR(s):
	# given string s (interval from value ranking), extract feature name, threshold(s) and logical operator
	# look for logical operators
	opslistins=[c for c in s if c in ['<','>','=']]
	finalops=[]
	for i in range(len(opslistins)):
		if opslistins[i]=='=' and i!=0:
			finalops.append(opslistins[i-1]+opslistins[i])
			del finalops[-2]
		else:
			finalops.append(opslistins[i])
	chforsplit=";"
	for e in finalops:
		s=s.replace(e,chforsplit)
	intervalelems = s.split(chforsplit)
	#print(intervalelems)
	if len(finalops) == 1:
		# single threshold interval
		feature_label = intervalelems[0].strip()
		threshold = float(intervalelems[1])
		return (feature_label, finalops[0],threshold)
	else:
		if len(finalops) == 2:
			# double threshold interval
			threshold1 = float(intervalelems[0])
			feature_label = intervalelems[1]
			threshold2 = float(intervalelems[2])
			return (threshold1,finalops[0],feature_label,finalops[1],threshold2)

def getValueRankingInfo(vrstringlist):
	# given list of value ranking intervals, returns a Nf (number of features) dimensional list of their elements
	intervals=[]
	for vrstring in vrstringlist:
		vrelements = ParseVR(vrstring)
		intervals.append(vrelements)
	#print(intervals)
	return intervals

def getFeatureLabels(intervals):
	flabels=[]
	for interval in intervals:
		if len(interval)==3:
			flabels.append(interval[0])
		else:
			if len(interval) == 5:
				flabels.append(interval[2])
	return flabels



def getOriginalThresholds(intervals):
	thresholds=[]
	for interval in intervals:
		if len(interval)==3:
			# singola soglia
			thresholds.append(interval[2])
		else:
			# doppia soglia
			if len(interval) == 5:
				thresholds.append((interval[0],interval[4]))
	return thresholds

def getOperators(intervals):
	operators=[]
	for interval in intervals:
		if len(interval)==3:
			# singola soglia
			operators.append(interval[1])
		else:
			# doppia soglia
			if len(interval) == 5:
				operators.append((interval[1],interval[3]))
	return operators




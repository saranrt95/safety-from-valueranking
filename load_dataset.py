import pandas as pd 
from config import *

def load_data(filename, class_label, flabels, minimizeFPR = False):			
	# filename: data filename
	# class_label: label of the output class
	# unsafe_class (optional): specifies the label of the class for the design of safety regions
	assert(filename[-4:]=='.csv' or filename[-5:]=='.xlsx' or filename[-4:]=='.txt')

	if filename[-4:]=='.csv':
		data=pd.read_csv(filename)
		datafeatures = data[flabels].copy()
		y_data=data[class_label]
		#data.drop([class_label], axis=1,inplace=True)
	else:
		if filename[-5:]=='.xlsx':
			data=pd.read_excel(filename)
			datafeatures = data[flabels].copy()
			y_data=data[class_label]
			#data.drop([class_label], axis=1,inplace=True)
		else:
			if filename[-4:]=='.txt':
				data=pd.read_csv(filename,delimiter="\t")
				datafeatures = data[flabels].copy()
				y_data=data[class_label]
				#data.drop([class_label], axis=1,inplace=True)
	
	if minimizeFPR:
		# switch '0' points with '1' --> capire se serve fare questo o altro
		#y_data = y_data.replace({0:1,1:0})
		pass
	
	return datafeatures,y_data
import pandas as pd
import numpy as np
from sympy.solvers import solve
from sympy import Symbol
import itertools as itr
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from safety_utils import *
class SafetyRegions():
		def __init__(self,filename,class_label,Nf, ths,fmin,fmax,cond_format,flabels,results_file,method,case, *pos_class):
			# filename: file with input data
			# Nf: number of features to be used (by now, only works with Nf=2)
			# class_label: name of output class
			# ths (list): original threshold values to be perturbed
			# fmin (list): minimum acceptable values for the features (order must be the same in ths)
			# fmax (list): maximum acceptable values for the features (order must be the same as in ths)
			# cond_format (list): 0 if perturbation is th-abs(delta*th), 1 otherwise
			# flabels: labels of the features to be perturbed
			# results_file (csv): output file with error, coverage, TP, TN, FP, FN for all the perturbation values (deltas)
			# method of the optimization algorithm: 'outside' or 'inside'
			# case ({1,2,3,4}): format of the intervals to be perturbed; 1 if f1>th1, f2<th2; 2 if f1<th1,f2>th2; 3 if f1>th1, f2>th2; 4 if f1<th1, f2<th2 
			# optional argument (*pos_class): if output variables are categorial, insert the label for the positive class
			self.filename=filename
			self.Nf=Nf
			self.class_label=class_label
			self.ths=ths
			self.fmin=fmin
			self.fmax=fmax
			self.cond_format=cond_format
			self.flabels=flabels
			self.results_file=results_file
			self.method=method
			self.case=case
			if (pos_class==None):
				self.pos_class=None
			else:
				self.pos_class=pos_class 


		#load data to be used for test, as pandas dataframe; class_label is the name of class column
		def load_data(self, filename, class_label):
			
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
			if self.pos_class!=None:
				#print(self.pos_class)
				y_data=y_data.replace(self.pos_class,1)
				y_data=y_data.where(y_data==1,0)
				#y_data[self.class_label]=self.pos_class]=le.transform(y_data)
			print(y_data)
			return (data,y_data)

		# get a csv file with error, coverage, TP,TN,FP,FN at varying of applied perturbations (delta1 and delta2)
		def get_all_metrics(self):
			test_set,y_test=self.load_data(self.filename,self.class_label)
			delta_lim=get_delta_limits(self.Nf,self.ths,self.fmin,self.fmax,self.cond_format)
			rule_perturbation(test_set,y_test,self.ths,self.flabels,delta_lim,self.results_file,self.method,self.case)

		# get the optimal perturbations (maximum TNR for minimum Error (hopefully 0))
		def get_optimal_results(self):
			self.get_all_metrics()
			output=pd.read_csv(self.results_file)
			minErr=min(output['Error'])
			outErrmin=output.loc[output['Error']==minErr]
			maxCov=max(outErrmin['Coverage'])
			outMaxCov=outErrmin.loc[outErrmin['Coverage']==maxCov]
			if self.method=='outside':
				print("Getting optimal results for {} -- outside method".format(self.filename))
				optdelta1=min(outMaxCov['Delta1'])
				optdelta2=min(outMaxCov['Delta2'])
				optimalResults=outMaxCov.loc[(outMaxCov['Delta1']==optdelta1) & (outMaxCov['Delta2']==optdelta2)]
			else:
				if self.method=='inside':
					print("Getting optimal results for {} -- inside method".format(self.filename))
					optdelta1=max(outMaxCov['Delta1'])
					optdelta2=max(outMaxCov['Delta2'])
					optimalResults=outMaxCov.loc[(outMaxCov['Delta1']==optdelta1) & (outMaxCov['Delta2']==optdelta2)]
			return optimalResults

		# scatter plot of the individuated safety regions
		def plot_safety_regions(self, outputRes):
			test_set,y_test=self.load_data(self.filename,self.class_label)
			test_set[self.class_label]=y_test
			th_new=[]
			for i in range(self.Nf):
				if self.cond_format[i]==0:
					th_new.append(self.ths[i]-abs(self.ths[i]*np.array(outputRes['Delta'+str(i+1)])))
				else:
					th_new.append(self.ths[i]+abs(self.ths[i]*np.array(outputRes['Delta'+str(i+1)])))
			print(th_new[0])
			print(th_new[1])
			fig, ax = plt.subplots()
			plt.scatter(test_set[self.flabels[0]].loc[test_set[self.class_label]==0],test_set[self.flabels[1]].loc[test_set[self.class_label]==0],marker='.')
			plt.scatter(test_set[self.flabels[0]].loc[test_set[self.class_label]==1],test_set[self.flabels[1]].loc[test_set[self.class_label]==1],marker='.')
			plt.legend(['safe (class=0)', 'unsafe (class=1)'])
			if self.method=='outside':
				if self.case==1:
					reg=ptc.Rectangle((self.fmin[0],float(th_new[1])),height=abs(self.fmax[1]-float(th_new[1])), width=abs(float(th_new[0])-self.fmin[0]), alpha=0.2, facecolor='magenta')
				else:
					if self.case==2:
						reg=ptc.Rectangle((float(th_new[0]), self.fmin[1]), height=abs(float(th_new[1])-self.fmin[1]), width=abs(self.fmax[0]-float(th_new[0])), alpha=0.2, facecolor='magenta')
					else:
						if self.case==3:
							reg=ptc.Rectangle((self.fmin[0], self.fmin[1]), height=abs(float(th_new[1])-self.fmin[1]), width=abs(float(th_new[0])-self.fmin[0]), alpha=0.2, facecolor='magenta')
						else:
							if self.case==4:
								reg=ptc.Rectangle((float(th_new[0]), float(th_new[1])),height=abs(self.fmax[1]-float(th_new[1])), width=abs(self.fmax[0]-float(th_new[0])), alpha=0.2, facecolor='magenta')
				ax.add_patch(reg)
			else:
				if self.method=='inside':
					if self.case==1:
						reg1=ptc.Rectangle((float(th_new[0]),self.fmin[1]), height=abs(self.fmax[1]-self.fmin[1]), width=abs(self.fmax[0]-float(th_new[0])), alpha=0.2, facecolor='magenta')
						reg2=ptc.Rectangle((self.fmin[0],self.fmin[1]), height=abs(float(th_new[1])-self.fmin[1]), width=abs(float(th_new[0])-self.fmin[0]), alpha=0.2, facecolor='magenta')
					else:
						if self.case==2:
							reg1=ptc.Rectangle((self.fmin[0],self.fmin[1]), height=abs(self.fmax[1]-self.fmin[1]), width=abs(float(th_new[0])-self.fmin[0]),alpha=0.2,facecolor='magenta')
							reg2=ptc.Rectangle((float(th_new[0]),float(th_new[1])), height=abs(self.fmax[1]-float(th_new[1])), width=abs(self.fmax[0]-float(th_new[0])),alpha=0.2,facecolor='magenta')
						else:
							if self.case==3:
								reg1=ptc.Rectangle((self.fmin[0],float(th_new[1])), height=abs(self.fmax[1]-float(th_new[1])), width=abs(self.fmin[0]-float(th_new[0])), alpha=0.2, facecolor='magenta')
								reg2=ptc.Rectangle((float(th_new[0]),self.fmin[1]), height=abs(self.fmax[1]-self.fmin[1]), width=abs(float(th_new[0])-self.fmax[0]), alpha=0.2, facecolor='magenta')
							else:
								if self.case==4:
									reg1=ptc.Rectangle((self.fmin[0],self.fmin[1]), height=abs(self.fmax[1]-self.fmin[1]), width=abs(self.fmin[0]-float(th_new[0])), alpha=0.2, facecolor='magenta')
									reg2=ptc.Rectangle((float(th_new[0]),self.fmin[1]), height=abs(float(th_new[1])-self.fmin[1]), width=abs(float(th_new[0])-self.fmax[0]), alpha=0.2, facecolor='magenta')
					ax.add_patch(reg1)
					ax.add_patch(reg2)
			ax.set_xticks([self.fmin[0],round(float(th_new[0]),2), self.fmax[0]])
			ax.set_yticks([self.fmin[1],round(float(th_new[1]),2), self.fmax[1]])
			plt.xlabel(self.flabels[0])
			plt.ylabel(self.flabels[1])
			plt.show()

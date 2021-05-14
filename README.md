# safety-from-valueranking
This repository contains the code and test example for the individuation of "safety regions" from rule-based models (Logic Learning Machine) feature and value ranking. By now, the focus is on binary classification problems and the shape of the obtained region is bidimensional (code works for Nf=2).
Two methods are available: 'outside' or 'inside';

# Usage

import safety_from_valueranking as sllm

It works with class labels in {0,1}.
safetyReg=sllm.SafetyRegions(filename,class_label,Nf,ths,fmin,fmax,cond_format,flabels,results_file,method,case)
For nominal labels, an optional argument is required (pos_class) to indicate the label of the positive class (will be encoded with 1):
safetyReg=sllm.SafetyRegions(filename,class_label,Nf,ths,fmin,fmax,cond_format,flabels,results_file,method,case, pos_class)

Input parameters:
      - filename: file with input data. 
			- Nf: number of features to be used (by now, only works with Nf=2). 
			- class_label: name of output class. 
			- ths (list): original threshold values to be perturbed. 
			- fmin (list): minimum acceptable values for the features (order must be the same in ths). 
			- fmax (list): maximum acceptable values for the features (order must be the same as in ths). 
			- cond_format (list): 0 if perturbation is th-abs(delta*th), 1 otherwise. 
			- flabels: labels of the features to be perturbed. 
			- results_file (csv): output file with error, coverage, TP, TN, FP, FN for all the perturbation values (deltas). 
			- method: method of the optimization algorithm: 'outside' or 'inside'. 
			- case ({1,2,3,4}): format of the intervals to be perturbed; 1 if f1>th1, f2<th2; 2 if f1<th1,f2>th2; 3 if f1>th1, f2>th2; 4 if f1<th1, f2<th2. 

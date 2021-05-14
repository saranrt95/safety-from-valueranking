# safety-from-valueranking
This repository contains the code and test example for the individuation of "safety regions" from rule-based models (e.g. Logic Learning Machine) feature and value ranking. By now, the focus is on binary classification problems and the shape of the obtained region is bidimensional (code works for Nf=2).
Two methods are available: 'outside' or 'inside'.
The two most relevant intervals obtained from a rule-based model can be tuned by our perturbation methods to achieve bidimensional regions denoted by the highest True Negative Rate (Coverage) being the False Negative Rate (Error) as close to 0 as possible.

# Usage
```
import safety_from_valueranking as sllm
```
It works with class labels in {0,1}.
```
safetyReg=sllm.SafetyRegions(filename,class_label,Nf,ths,fmin,fmax,cond_format,flabels,results_file,method,case). 
```
For nominal labels, an optional argument is required (pos_class) to indicate the label of the positive class (will be encoded with 1):  
```
safetyReg=sllm.SafetyRegions(filename,class_label,Nf,ths,fmin,fmax,cond_format,flabels,results_file,method,case, pos_class)
```
Input parameters:
- filename: file with input data. 
- Nf: number of features to be used (by now, only works with Nf=2). 
- class_label: name of output class. 
- ths (Nf-dim list of float): original threshold values to be perturbed. 
- fmin (Nf-dim list of float): minimum acceptable values for the features (order must be the same in ths). 
- fmax (Nf-dim list of float): maximum acceptable values for the features (order must be the same as in ths). 
- cond_format (Nf-dim list of int): 0 or 1, indicates the kind of perturbation (see below) 
- flabels (list of strings) : labels of the features to be perturbed. 
- results_file (csv): output file with error, coverage, TP, TN, FP, FN for all the perturbation values (Delta1,Delta2). 
- method (string): method of the optimization algorithm: 'outside' or 'inside'. 
- case (int in {1,2,3,4}): format of the pair of intervals to be perturbed;
- optional: pos_class

The software considers perturbations on single-threshold intervals for the 2 features (f1 and f2), that can have different formats. The information about this is needed by defining the case input variable as follows:
FORMAT OF THE INTERVALS TO BE PERTURBED | CASE 
----------------------------------------|-----  
f1 >= original_threshold1 and f2 <= original_threshold2	|1
f1 <= original_threshold1 and f2 >= original_threshold2 |2
f1 >= original_threshold1 and f2 >= original_threshold2 |3
f1 <= original_threshold1 and f2 <= original_threshold2 |4


Useful instructions for cond_format:

METHOD | FORMAT OF THE INTERVAL TO BE PERTURBED | COND_FORMAT
-------| ---------------------------------------|------------  
'outside'| feature >= original_threshold| 0
'outside'| feature <= original_threshold| 1
'inside' | feature >= original_threshold| 1
'inside' | feature <= original_threshold| 0

The results can be obtained by calling:
```
output=safetyReg.get_optimal_results()
```
It will print the optimal perturbations applied (Delta1, Delta2), along with the FNR, TNR, TP,TN,FP and FN metrics.
The methods are based on iterations over candidate perturbations. For this reason, an output file with all the collected metrics during iterations is produced.

Finally, you can plot the obtained "safety regions" by calling:
```
safetyReg.plot_safety_regions(output)
```

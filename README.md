# safety-from-valueranking
This repository contains the code and test examples described in the following papers (please cite if you use the code):

Narteni, S., Ferretti, M., Orani, V., Vaccari, I., Cambiaso, E., Mongelli, M. (2021). From Explainable to Reliable Artificial Intelligence. In: Holzinger, A., Kieseberg, P., Tjoa, A.M., Weippl, E. (eds) Machine Learning and Knowledge Extraction. CD-MAKE 2021. Lecture Notes in Computer Science(), vol 12844. Springer, Cham. https://doi.org/10.1007/978-3-030-84060-0_17 [Skope-Rules tests have been presented only in this conference paper];

S. Narteni, V. Orani, I. Vaccari, E. Cambiaso and M. Mongelli, "Sensitivity of Logic Learning Machine for reliability in safety-critical systems," in IEEE Intelligent Systems, doi: 10.1109/MIS.2022.3159098.



The aim is the individuation of "safety regions" from rule-based models (e.g. Logic Learning Machine) feature and value ranking. By now, the focus is on binary classification problems and the shape of the obtained region is bidimensional (code works for Nf=2).
Two methods are available: 'outside' or 'inside'.
The two most relevant intervals obtained from a rule-based model can be tuned by our perturbation methods to achieve bidimensional regions denoted by the highest True Negative Rate (Coverage) being the False Negative Rate (Error) as close to 0 as possible.

In addition, LLM with 0% error can be used to look for more complex "safety regions" with higher dimensions (>3 features).
# Usage for Outside and Inside Methods
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
# Example
Examples of usage are provided in test.py (by now, 'outside' method for physical fatigue data is selected, but you can deactivate comments from other test cases).

# Usage for LLM 0% and Skope-Rules

Just run the scripts ("platooning-LLMzero.py", "fatigue_zeroerror.py", "PlatooningSkopeZeroError.py", "FatigueSkopeZeroError.py") and the optimal solution will be printed.

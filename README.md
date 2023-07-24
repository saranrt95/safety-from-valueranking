# safety-from-valueranking
This repository contains the code and test examples described in the following papers (please cite if you use the code):

Narteni, S., Ferretti, M., Orani, V., Vaccari, I., Cambiaso, E., Mongelli, M. (2021). From Explainable to Reliable Artificial Intelligence. In: Holzinger, A., Kieseberg, P., Tjoa, A.M., Weippl, E. (eds) Machine Learning and Knowledge Extraction. CD-MAKE 2021. Lecture Notes in Computer Science(), vol 12844. Springer, Cham. https://doi.org/10.1007/978-3-030-84060-0_17 [Skope-Rules tests have been presented only in this conference paper];

S. Narteni, V. Orani, I. Vaccari, E. Cambiaso and M. Mongelli, "Sensitivity of Logic Learning Machine for reliability in safety-critical systems," in IEEE Intelligent Systems, doi: 10.1109/MIS.2022.3159098.



The aim is the individuation of "safety regions" from rule-based models (e.g. Logic Learning Machine) feature and value ranking. 
Two methods are available: 'outside' or 'inside'.
When working in two dimensions, the two most relevant intervals obtained from a rule-based model can be tuned by our perturbation methods to achieve bidimensional regions denoted by the highest True Negative Rate (Coverage) being the False Negative Rate (Error) as close to 0 as possible.

In addition, LLM with 0% error can be used to look for more complex "safety regions" with higher dimensions (>3 features).
# Usage for Outside and Inside Methods

To reproduce the experiments of the papers or use the methods for your own, just set the required inputs in the `config.py' file. Then, just run `main.py' and the results will be automatically generated.


# Usage for LLM 0% and Skope-Rules

Just run the scripts ("platooning-LLMzero.py", "fatigue_zeroerror.py", "PlatooningSkopeZeroError.py", "FatigueSkopeZeroError.py") and the optimal solution will be printed.

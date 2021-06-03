import safety_from_valueranking as sllm

'''
filename="platooning_test.xlsx"
class_label='collision'
Nf=2 
ths=[0.43, -7.50]
fmin=[0.2, -8]
fmax=[0.5, -1]
cond_format=[0,1]
flabels=['PER','F0']
results_file="vp_outside_new.csv"
method='outside'
case=1
'''
'''
filename="platooning_test.xlsx"
class_label='collision'
Nf=2 
ths=[0.33, -3.50]
fmin=[0.2, -8]
fmax=[0.5, -1]
cond_format=[0,1]
flabels=['PER','F0']
results_file="vp_inside_new.csv"
method='inside'
case=2
'''

filename="MMH_test_set_nomin.xlsx"
class_label='fatiguestate1'
Nf=2 # N.B. per ora non funziona con più di 2
ths=[0.03, 0.03]
fmin=[-1.86,-2]#-1.78]
fmax=[3.12, 3.95]
cond_format=[0,0]
flabels=['back rotation position in sag plane','Wrist.jerk.coefficient.of.variation']
results_file="mmh_outside_new.csv"
method='outside'
case=3

'''
filename="MMH_test_set.xlsx"
class_label='fatiguestate1'
Nf=2 
ths=[0.03, -0.47]
fmin=[-1.86, -1.16]
fmax=[3.12, 3.99]
cond_format=[0,1]
flabels=['back rotation position in sag plane','Chest.ACC.Mean']
results_file="mmh_inside_new.csv"
method='inside'
case=2
'''
if __name__ == '__main__':
	safetyReg=sllm.SafetyRegions(filename,class_label,Nf,ths,fmin,fmax,cond_format,flabels,results_file,method,case)
	#safetyReg.get_all_metrics()
	output=safetyReg.get_optimal_results()
	#output.to_excel('optres.xlsx')
	print(output)
	safetyReg.plot_safety_regions(output)

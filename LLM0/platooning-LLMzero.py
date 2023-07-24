import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itr

test_set=pd.read_excel('platooning_test.xlsx')
y_test=test_set['collision']
test_set.drop(['collision'], axis = 1, inplace = True)
feature_labels=['N','F0','PER','d0','v0']

#trovati da disequazione
deltaPER_max=0.322033898305085
deltaPER_min=-0.322033898305085
deltaPER_range=np.arange(deltaPER_min,deltaPER_max,0.05)
deltaV0_min=-0.649122807017544
deltaV0_max=0.649122807017544
deltaV0_range=np.arange(deltaV0_min, deltaV0_max,0.05)

with open('platooning-LLMzero.csv','a') as output:
    output.write("Delta1"+","+"Delta2"+","+"Error"+","+"Coverage"+","+"TP, FP, TN, FN"+"\n")

for deltaPER,deltaV0 in itr.product(deltaPER_range,deltaV0_range):
        TP=0
        FP=0
        FN=0
        TN=0
        predicted=0
        for i in range(0,len(test_set)):
             
             # OR delle 4 regole LLM a zero error per la classe 0 (no collisioni); quando stessa feature compare su più regole,
             # perturbo solo la condizione più restrittiva
             if (( test_set['N'][i]<=5 and test_set['v0'][i]<=54.50)
              or (test_set['PER'][i]<=0.295-abs(0.295*deltaPER) and test_set['N'][i]<=7 and test_set['v0'][i]<=86.50)
              #or (test_set['v0'][i]<=27.50)
              or (test_set['v0'][i]<=28.50-abs(28.50*deltaV0) and test_set['PER'][i]<=0.4455)
              or (test_set['v0'][i]<=38.50 and test_set['N'][i]<=6 and test_set['d0'][i]<= 7.8615)
             ):
                 predicted=0
             else:
                 predicted=1 
             #print((predicted, y_test[i]))
             if (predicted==1 and y_test[i]==1):
                 #print("true positive")
                 TP+=1
             else: 
                 if (predicted==1 and y_test[i]==0):
                        #print("false positive") 
                        FP+=1
                 else: 
                        if (predicted==0 and y_test[i]==0):
                               #print("true negative") 
                               TN+=1
                        else:
                               if (predicted==0 and y_test[i]==1):
                                      #print("false negative")
                                      FN+=1

        #print("TP: "+str(TP))
        #print("FP: "+str(FP))
        #print("FN: "+str(FN))
        #print("TN: "+str(TN))

        error=FN/(FN+TP) # false negative rate
        #print("Error: "+str(error))
        coverage=TN/(TN+FP) #covering per la classe 0 (TNR, misura la size della regione "non fatigued")
        #print("Coverage: "+str(coverage))
        with open('platooning-LLMzero.csv','a') as output:
            output.write(str(deltaPER)+","+str(deltaV0)+","+str(error)+","+str(coverage)+","+str(TP)+','+str(FP)+','+str(TN)+','+str(FN)+'\n')



output=pd.read_csv('platooning-LLMzero.csv')
minErr=min(output['Error'])
outErrmin=output.loc[output['Error']==minErr]
maxCov=max(outErrmin['Coverage'])
outMaxCov=outErrmin.loc[outErrmin['Coverage']==maxCov]
print("Getting optimal results for {} -- LLM0% method".format('platooning_test.xlsx'))
optdelta1=max(outMaxCov['Delta1'])
optdelta2=max(outMaxCov['Delta2'])
optimalResults=outMaxCov.loc[(outMaxCov['Delta1']==optdelta1) & (outMaxCov['Delta2']==optdelta2)]
print(optimalResults)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
import math
'''
# LOAD DATA
data = pd.read_csv("MMH_15p.csv",index_col=0) 
# Preview the first 5 lines of the loaded data 
#print(data)
data.drop(['task','gender','fatiguestate','subject','HRR-Mean','HRR-CV'], axis = 1, inplace = True) # per includere subject e unnamed (indice dei record)
# GET CLASS LABELS
target = data['fatiguestate1']
data.drop(['fatiguestate1'], axis = 1, inplace = True)
target = np.asarray(target)

feature_labels=['age', 'Wrist.jerk.Mean', 'Wrist.ACC.Mean', 'Hip.jerk.Mean',
       'Hip.ACC.Mean', 'Hip.xposture.Mean', 'Hip.yposture.Mean',
       'Hip.zposture.Mean', 'Chest.jerk.Mean', 'Chest.ACC.Mean',
       'Chest.xposture.Mean', 'Chest.yposture.Mean', 'Chest.zposture.Mean',
       'Ankle.jerk.Mean', 'Ankle.ACC.Mean', 'Ankle.xposture.Mean',
       'number of steps', 'average step time', 'average step distance',
       'time bent', 'average back bent angle', 'mean hip osicllation',
       'mean foot osicllation', 'leg rotational velocity sag plane',
       'leg rotational position sag plane', 'average vertical impact',
       'back rotation position in sag plane',
       'Wrist.jerk.coefficient.of.variation',
       'Wrist.ACC.coefficient.of.variation',
       'Hip.jerk.coefficient.of.variation', 'Hip.ACC.coefficient.of.variation',
       'Chest.jerk.coefficient.of.variation',
       'Chest.ACC.coefficient.of.variation',
       'Ankle.jerk.coefficient.of.variation',
       'Ankle.ACC.coefficient.of.variation',
       'Hip.yposture.coefficient.of.variation',
       'Chest.yposture.coefficient.of.variation',
       'Ankle.yposture.coefficient.of.variation']

#print(data['gender'])

# SPLIT DATA (incluso subject)
X_train_t, X_test_t, y_train, y_test = train_test_split(data.values, target, test_size=0.33, random_state=42) 

# # Salvo a parte subjects e index e li rimuovo da X_train_t e X_test
# subjects_train=[]
# subjects_test=[]
# index_train=[]
# index_test=[]
# for e in X_train_t:
#   index_train.append(e[0])
#   subjects_train.append(e[1]) #copy subject label
# X_train_t=np.delete(X_train_t,[0,1],1)
# for e in X_test_t:
#   index_test.append(e[0])
#   subjects_test.append(e[1])#copy subject label
# X_test_t=np.delete(X_test_t,[0,1],1)

# print(np.shape(X_train_t))
# print(np.shape(X_test_t))
# print(X_test_t)
# print(X_train_t)

# STANDARDADIZZAZIONE 
scaler = StandardScaler()
X_train_t = scaler.fit_transform(X_train_t)
X_test_t = scaler.fit_transform(X_test_t)

#print(X_test_t)
#print(X_train_t)

#codice usato per salvare  test set
test_set=pd.DataFrame(X_test_t,columns=feature_labels)
test_set.insert(0,'fatiguestate1',y_test)
print(test_set)
#test_set.to_excel("MMH_test_set.xlsx")

train_set=pd.DataFrame(X_train_t,columns=feature_labels)
train_set.insert(0,'fatiguestate1',y_train)
print(train_set)
train_set.to_excel("MMH_training_set.xlsx")

'''
# DATASET 
X_test_t=pd.read_excel("MMH_test_set.xlsx")
y_test=X_test_t['fatiguestate1']
X_test_t.drop(['fatiguestate1'], axis = 1, inplace = True)

increm_delta=0.05

with open("output_LLM0_fatigue.csv",'a') as output:
       output.write("Delta1"+","+"Delta2"+","+"Error"+","+"Coverage"+","+"TP, FP, TN, FN"+"\n")
for delta1 in np.arange(0.048,1.88,increm_delta):
    for delta2 in np.arange(-1.623,5.043,increm_delta):

      TP=0
      FP=0
      FN=0
      TN=0
      predicted=0
      for i in range(0,len(X_test_t)):

             # NO PERTURBAZIONI
             # if ( (0.507675 < X_test_t['Hip.ACC.Mean'][i]<=1.985149 and X_test_t['Chest.ACC.coefficient.of.variation'][i]<= 1.110217 and -1.429425 < X_test_t['Wrist.ACC.coefficient.of.variation'][i]  <= 1.499825 and -1.732093 < X_test_t['average step distance'][i] <= 0.813419 and X_test_t['back rotation position in sag plane'][i]<=0.522655)
             # or (X_test_t['Wrist.jerk.Mean'][i] > 0.549022  and -1.346784 < X_test_t['back rotation position in sag plane'][i]<=0.042555)
             # or (-1.732093 < X_test_t['average step distance'][i] <= -0.229824 and -0.446093 < X_test_t['number of steps'][i] <= 3.754398 and -1.734997 < X_test_t['Wrist.jerk.coefficient.of.variation'][i]<= 0.557292 and X_test_t['back rotation position in sag plane'][i]<=-0.255161 )
             # or (X_test_t['Chest.xposture.Mean'][i] > -0.033129  and  X_test_t['Hip.zposture.Mean'][i] >0.433513  and X_test_t['Wrist.ACC.Mean'][i] > -0.832691 and -0.885459 < X_test_t['back rotation position in sag plane'][i]<= 0.291992)
             # or (( -1.647118 < X_test_t['Hip.yposture.coefficient.of.variation'][i] <= -0.670325) and  (X_test_t['leg rotational velocity sag plane'][i]> 0.590344 ))
             # or ( -0.648650 < X_test_t['Hip.jerk.Mean'][i]<= 1.146078 and X_test_t['Ankle.xposture.Mean'][i] <= -1.478901 )
             # ):
             # Con perturbazione prime 2
             if ( (0.507675 < X_test_t['Hip.ACC.Mean'][i]<=1.985149 - delta1*1.985149 and X_test_t['Chest.ACC.coefficient.of.variation'][i]<= 1.110217 and -1.429425 < X_test_t['Wrist.ACC.coefficient.of.variation'][i]  <= 1.499825 and -1.732093 < X_test_t['average step distance'][i] <= 0.813419 and X_test_t['back rotation position in sag plane'][i]<=0.522655)
             or (X_test_t['Wrist.jerk.Mean'][i] > 0.549022 + delta2 * 0.549022 and -1.346784 < X_test_t['back rotation position in sag plane'][i]<=0.042555)
             or (-1.732093 < X_test_t['average step distance'][i] <= -0.229824 and -0.446093 < X_test_t['number of steps'][i] <= 3.754398 and -1.734997 < X_test_t['Wrist.jerk.coefficient.of.variation'][i]<= 0.557292 and X_test_t['back rotation position in sag plane'][i]<=-0.255161 )
             or (X_test_t['Chest.xposture.Mean'][i] > -0.033129  and  X_test_t['Hip.zposture.Mean'][i] >0.433513  and X_test_t['Wrist.ACC.Mean'][i] > -0.832691 and -0.885459 < X_test_t['back rotation position in sag plane'][i]<= 0.291992)
             #or (( -1.647118 < X_test_t['Hip.yposture.coefficient.of.variation'][i] <= -0.670325) and  (X_test_t['leg rotational velocity sag plane'][i]> 0.590344 ))
             #or ( -0.648650 < X_test_t['Hip.jerk.Mean'][i]<= 1.146078 and X_test_t['Ankle.xposture.Mean'][i] <= -1.478901 )
             ):
                 predicted=0 #rule applies
             else:
                 predicted=1 #rule doesn't apply
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
      coverage=TN/(TN+FP) #covering per la classe 0 (misura la size della regione "non fatigued")
      #print("Coverage: "+str(coverage))
      #if error<=0.01 and coverage>0: # escludo coverage = 0 (regioni vuote)
      #print("Error: "+str(error))
      #print("Coverage: "+str(coverage))
      #print((delta1,delta2,delta3))
      with open("output_LLM0_fatigue.csv",'a') as output:
          output.write(str(delta1)+","+str(delta2)+","+str(error)+","+str(coverage)+','+str(TP)+','+str(FP)+','+str(TN)+','+str(FN)+'\n')

output=pd.read_csv('output_LLM0_fatigue.csv')
minErr=min(output['Error'])
outErrmin=output.loc[output['Error']==minErr]
maxCov=max(outErrmin['Coverage'])
outMaxCov=outErrmin.loc[outErrmin['Coverage']==maxCov]
print("Getting optimal results for {} -- LLM0% method".format('MMH_test_set.xlsx'))
optdelta1=max(outMaxCov['Delta1'])
optdelta2=max(outMaxCov['Delta2'])
optimalResults=outMaxCov.loc[(outMaxCov['Delta1']==optdelta1) & (outMaxCov['Delta2']==optdelta2)]
print(optimalResults)
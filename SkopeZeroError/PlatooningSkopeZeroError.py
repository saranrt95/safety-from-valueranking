#Install skope-rules
pip install skope-rules

# Import skope-rules
from skrules import SkopeRules


# Import librairies
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib import cm
import numpy as np
from sklearn.metrics import confusion_matrix
from IPython.display import display

#definition of all function used
def non_collisioni (row):
   if row['collision'] == 0 :
      return 1
   else:
      return 0

def TrovaNumeroGiusto(max_depth_duplication, n_estimators):
  global Confronti
  skope_rules = SkopeRules(feature_names=feature_labels, random_state=42, n_estimators=n_estimators,
                               recall_min=0.05, precision_min=1,
                               max_samples=0.7,
                               max_depth_duplication= max_depth_duplication, max_depth = 5)
  skope_rules.fit(training_set, y_trainNC)
  for n_rule_chosen in range(len(skope_rules.rules_)):
    y_predNC = skope_rules.predict_top_rules(test_set, n_rule_chosen+1)
    precision_recall = compute_performances_from_y_pred(y_testNC, y_predNC, 'test_set')
    new_row = {'recall_min':0.05,
              'precision_min':0.9,
              'n_estimators': n_estimators,
              'max_depth_duplication': max_depth_duplication,
              'precision': precision_recall.iloc[0, 0],
              'recall': precision_recall.iloc[0, 1],
              'numero di regole': n_rule_chosen+1
             }
    Confronti = Confronti.append(new_row, ignore_index=True)

def compute_performances_from_y_pred(y_true, y_pred, index_name='default_index'):
    df = pd.DataFrame(data=
        {
            'precision':[sum(y_true * y_pred)/sum(y_pred)],
            'recall':[sum(y_true * y_pred)/sum(y_true)]
        },
        index=[index_name],
        columns=['precision', 'recall']
    )
    return(df)

def calcolaCollision_previsto (df, funzione, deltaPER, deltaN):
  if 'Collision_previsto' in df.columns:
    del df['Collision_previsto']
  df['Collision_previsto'] = df.apply (lambda row: funzione(row, deltaPER, deltaN), axis=1)

#Regole 0 error
def regole(row, delta1, delta2):
  if ((row['PER'] <= 0.41499999165534973 and row['v0']<= 45.5-abs(45.5*delta2)) or
      (row['N'] <= 7.5 and row['F0']> -7.5 and row['PER']<= 0.32500000298023224-abs(0.32500000298023224*delta1)) or
      (row['N']<= 5.5 and row['v0']<= 54.5) or
      (row['F0']> -4.5  and row['PER']<= 0.41499999165534973 and row['v0']> 64.5)
      ):
        return 0
  else:
        return 1

def Statistiche (df, delta1=0, delta2=0):
  global Confronti
  colonna = 'Collision_previsto'
  calcolaCollision_previsto (df, regole, delta1, delta2)
  new_row = {'Dimensione_dataset':df.shape[0],
             'Totale_collisioni':df.loc[(df['collision'] == 1)]['collision'].count(),
             'Totale_non_collisioni':df.loc[(df['collision'] == 0)]['collision'].count(),
             'Collision_previsto': df.loc[(df[colonna] == 1)][colonna].count(),
             'deltaPER': delta1,
             'deltaV0': delta2,
             'TP': df.loc[(df['collision'] == 1) & (df[colonna]==1)]['collision'].count(),
             'FP': df.loc[(df['collision'] == 0) & (df[colonna]==1)]['collision'].count(),
             'TN': df.loc[(df['collision'] == 0) & (df[colonna]==0)]['collision'].count(),
             'FN': df.loc[(df['collision'] == 1) & (df[colonna]==0)]['collision'].count(),
             'error' : (df.loc[(df['collision'] == 1) & (df[colonna]==0)]['collision'].count())/
                       (df.loc[(df['collision'] == 1) & (df[colonna]==0)]['collision'].count() +
                        df.loc[(df['collision'] == 1) & (df[colonna]==1)]['collision'].count()),  # error=FN/(FN+TP)
             'coverage' : (df.loc[(df['collision'] == 0) & (df[colonna]==0)]['collision'].count())/
                          (df.loc[(df['collision'] == 0) & (df[colonna]==0)]['collision'].count()+
                           df.loc[(df['collision'] == 0) & (df[colonna]==1)]['collision'].count()),# =TN/(TN+FP)
             'Precision':  (df.loc[(df['collision'] == 0) & (df[colonna]==0)]['collision'].count())/
               ((df.shape[0])-df.loc[(df[colonna] == 1)][colonna].count()), # Precision = VereNonCollisioniPreviste/NonCollisioniPreviste
             'Recall': (df.loc[(df['collision'] == 0) & (df[colonna]==0)]['collision'].count())/
               (df.loc[(df['collision'] == 0)]['collision'].count()), # recall = VereNonCollisioniPreviste/TotaleNonCollisioni
             'FNR': (df.loc[(df['collision'] == 1) & (df[colonna]==0)]['collision'].count())/
               (df.loc[(df['collision'] == 1)]['collision'].count()), # FNR = Falsi negativi / Condizioni positive
             'TNR': (df.loc[(df['collision'] == 0) & (df[colonna]==0)]['collision'].count())/
                    (df.loc[(df['collision'] == 0)]['collision'].count()),#TNR = TN ÷ N
             'FScore': 2*(df.loc[(df['collision'] == 1) & (df[colonna]==1)]['collision'].count())/
               (2*(df.loc[(df['collision'] == 1) & (df[colonna]==1)]['collision'].count())+
               (df.loc[(df['collision'] == 1) & (df[colonna]==0)]['collision'].count())+
               (df.loc[(df['collision'] == 0) & (df[colonna]==1)]['collision'].count())) # Fscore
            }
  Confronti = Confronti.append(new_row, ignore_index=True)


# Load the platooning dataset
platooning_test = pd.read_excel("platooning_test.xlsx")
platooning_training = pd.read_excel("platooning_training.xlsx")

#non_collision column creation to train the classifier to find non-collision rules
platooning_test['non_collision'] = platooning_test.apply (lambda row: non_collisioni(row), axis=1)
platooning_training['non_collision'] = platooning_training.apply (lambda row: non_collisioni(row), axis=1)
y_testNC = platooning_test['non_collision']
y_trainNC = platooning_training['non_collision']
training_set = platooning_training.drop(['collision', 'non_collision'], axis=1)
test_set = platooning_test.drop(['collision', 'non_collision'], axis=1)
feature_labels = training_set.columns

#Run the algorithm by varying max_depth_duplication and n_estimators to choose the best rules. To do this
#create a matrix which is the vector product of the different possible choices of max_depth_duplication and n_estimators.
#Finally, calculate the precision and recall parameters for each model and save everything on a .csv and then choose which model to use.
columns = ['recall_min', 'precision_min', 'n_estimators', 'max_depth_duplication', 'precision',
           'recall', 'numero di regole']
Confronti = pd.DataFrame(columns=columns)

n_estimators_min=25
n_estimators_max=210
n_estimators_range=np.arange(n_estimators_min,n_estimators_max,25)
max_depth_duplication_min=2
max_depth_duplication_max=6
max_depth_duplication_range=np.arange(max_depth_duplication_min, max_depth_duplication_max,1)

y = n_estimators_range
x = max_depth_duplication_range
xx, yy = np.meshgrid(x, y)
xx, yy
m = np.zeros((len(x)*len(y), 2))
indice = 0
for i in range(len(xx)):
  for j in range(len(x)):
    #print('i: ' +str(i) + ' j: ' +str(j) + ' indice: ' +str(indice))
    m[indice, 0] = xx[i , j]
    indice = indice + 1
indice = 0
for i in range(len(yy)):
  for j in range(len(x)):
    m[indice, 1] = yy[i , j]
    indice = indice + 1

for i in range(len(m)):
    TrovaNumeroGiusto(int(m[i, 0]), int(m[i, 1]))
Confronti.to_csv('ConfrontiPlatooningZE.csv', index=False)

#train the model with the parameters that seemed optimal
n_estimators = 75
max_depth_duplication = 2
skope_rules = SkopeRules(feature_names=feature_labels, random_state=42, n_estimators=n_estimators,
                               recall_min=0.05, precision_min=1,
                               max_samples=0.7,
                               max_depth_duplication= max_depth_duplication, max_depth = 5)
skope_rules.fit(training_set, y_trainNC)

# Visualization of the rules
for i_rule, rule in enumerate(skope_rules.rules_):
    print(rule[0])

#Try out the selected rules by perturbing two features.
#Create a matrix which is the vector product of the perturbations of the two features.
#Calculate the precision and recall parameters for each pair of perturbations and save everything to .csv in order to choose the perturbation that best approximates zero error.
columns = ['Dimensione_dataset', 'Totale_collisioni', 'Totale_non_collisioni', 'Collision_previsto',
           'deltaPER', 'deltaV0', 'TP', 'FP', 'TN', 'FN', 'error', 'coverage', 'Precision', 'Recall', 'FNR', 'TNR', 'FScore']
Confronti = pd.DataFrame(columns=columns)

deltaPER_max=0.322033898305085
deltaPER_min=-0.322033898305085
deltaPER_range=np.arange(deltaPER_min,deltaPER_max,0.05)
deltaV0_min=-0.649122807017544
deltaV0_max=0.649122807017544
deltaV0_range=np.arange(deltaV0_min, deltaV0_max,0.05)

y = deltaPER_range
x = deltaV0_range
xx, yy = np.meshgrid(x, y)
xx, yy
m = np.zeros((len(x)*len(y), 2))
indice = 0
for i in range(len(xx)):
  for j in range(len(x)):
    m[indice, 0] = xx[i , j]
    indice = indice + 1
indice = 0
for i in range(len(yy)):
  for j in range(len(x)):
    m[indice, 1] = yy[i , j]
    indice = indice + 1

Statistiche (platooning_test, 0, 0)
for i in range(len(m)):
    Statistiche (platooning_test, m[i, 1], m[i, 0])
Confronti.to_csv('ConfrontiNE.csv', index=False)
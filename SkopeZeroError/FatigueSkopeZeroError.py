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
def non_fatica (row):
   if row['fatiguestate1'] == 0 :
      return 1
   else:
      return 0

def TrovaNumeroGiusto(max_depth_duplication, n_estimators):
  global Confronti
  skope_rules = SkopeRules(feature_names=feature_labels, random_state=42, n_estimators=n_estimators,
                               recall_min=0.05, precision_min=1,
                               max_samples=0.7,
                               max_depth_duplication= max_depth_duplication, max_depth = 5)
  skope_rules.fit(training_set, y_trainNF)
  for n_rule_chosen in range(len(skope_rules.rules_)):
    y_predNF = skope_rules.predict_top_rules(test_set, n_rule_chosen+1)
    precision_recall = compute_performances_from_y_pred(y_testNF, y_predNF, 'test_set')
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

def calcolaFatigue_previsto (df, funzione, delta1, delta2):
  if 'Fatigue_previsto' in df.columns:
    del df['Fatigue_previsto']
  df['Fatigue_previsto'] = df.apply (lambda row: funzione(row, delta1, delta2), axis=1)

def regole (row, delta1, delta2):
  if (( row['back rotation position in sag plane'] <= 0.08208675496280193 - delta1 * 0.08208675496280193 and
        row['Hip.jerk.Mean'] > -1.0291672945022583 and
        row['Hip.ACC.coefficient.of.variation'] <= 0.7530215680599213 and
        row['Hip.yposture.Mean'] <= 1.11860853433609 and
        row['Hip.zposture.Mean'] > -1.7837491035461426 ) or
      ( row['back rotation position in sag plane'] <= 0.17566733807325363 and
        row['Wrist.jerk.coefficient.of.variation'] <= 0.05163064785301685 - delta2 * 0.05163064785301685 and
        row['Hip.ACC.Mean'] > -0.47386549413204193 ) or
      ( row['back rotation position in sag plane'] <= 0.22119303047657013 and
        row['Wrist.jerk.coefficient.of.variation'] <= 0.05529268458485603 and
        row['Hip.ACC.Mean'] > -0.09642184898257256 and
        row['Chest.jerk.Mean'] > -1.357083261013031)
      ):
        return 0
  else:
        return 1

def Statistiche (df, delta1=0, delta2=0):
  global Confronti
  colonna = 'Fatigue_previsto'

  calcolaFatigue_previsto (df, regole, delta1, delta2)

  new_row = {'Dimensione_dataset':df.shape[0],
             'Totale_affaticati':df.loc[(df['fatiguestate1'] == 1)]['fatiguestate1'].count(),
             'Totale_non_affaticati':df.loc[(df['fatiguestate1'] == 0)]['fatiguestate1'].count(),
             'Fatigue_previsto': df.loc[(df[colonna] == 1)][colonna].count(),
             'delta1': delta1,
             'delta2': delta2,
             'TP': df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==1)]['fatiguestate1'].count(),
             'FP': df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==1)]['fatiguestate1'].count(),
             'TN': df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==0)]['fatiguestate1'].count(),
             'FN': df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==0)]['fatiguestate1'].count(),
             'error' : (df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==0)]['fatiguestate1'].count())/
                       (df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==0)]['fatiguestate1'].count() +
                        df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==1)]['fatiguestate1'].count()),  # error=FN/(FN+TP)
             'coverage' : (df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==0)]['fatiguestate1'].count())/
                          (df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==0)]['fatiguestate1'].count()+
                           df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==1)]['fatiguestate1'].count()),# =TN/(TN+FP)
             'Precision':  (df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==0)]['fatiguestate1'].count())/
               ((df.shape[0])-df.loc[(df[colonna] == 1)][colonna].count()), # Precision = VereNonAffaticatiPreviste/NonAffaticatiPreviste
             'Recall': (df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==0)]['fatiguestate1'].count())/
               (df.loc[(df['fatiguestate1'] == 0)]['fatiguestate1'].count()), # recall = VereNonAffaticatePreviste/TotaleNonAffaticati
             'FNR': (df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==0)]['fatiguestate1'].count())/
               (df.loc[(df['fatiguestate1'] == 1)]['fatiguestate1'].count()), # FNR = Falsi negativi / Condizioni positive
             'TNR': (df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==0)]['fatiguestate1'].count())/
                    (df.loc[(df['fatiguestate1'] == 0)]['fatiguestate1'].count()),#TNR = TN ÷ N
             'FScore': 2*(df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==1)]['fatiguestate1'].count())/
               (2*(df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==1)]['fatiguestate1'].count())+
               (df.loc[(df['fatiguestate1'] == 1) & (df[colonna]==0)]['fatiguestate1'].count())+
               (df.loc[(df['fatiguestate1'] == 0) & (df[colonna]==1)]['fatiguestate1'].count())) # Fscore
            }
  Confronti = Confronti.append(new_row, ignore_index=True)


# Load the fatigue dataset
MMH_test_set = pd.read_excel("MMH_test_set.xlsx")
MMH_training_set = pd.read_excel("MMH_training_set.xlsx")
MMH_test_set = MMH_test_set.drop(['Unnamed: 0'], axis=1)
MMH_training_set = MMH_training_set.drop(['Unnamed: 0'], axis=1)

#non_fatuguestate1 column creation to train the classifier to find non-fatigue rules
MMH_test_set['non_fatiguestate1'] = MMH_test_set.apply (lambda row: non_fatica(row), axis=1)
MMH_training_set['non_fatiguestate1'] = MMH_training_set.apply (lambda row: non_fatica(row), axis=1)

y_testNF = MMH_test_set['non_fatiguestate1']
y_trainNF = MMH_training_set['non_fatiguestate1']
training_set = MMH_training_set.drop(['fatiguestate1', 'non_fatiguestate1'], axis=1)
test_set = MMH_test_set.drop(['fatiguestate1', 'non_fatiguestate1'], axis=1)
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
    m[indice, 0] = xx[i , j]
    indice = indice + 1
indice = 0
for i in range(len(yy)):
  for j in range(len(x)):
    m[indice, 1] = yy[i , j]
    indice = indice + 1

for i in range(len(m)):
    TrovaNumeroGiusto(int(m[i, 0]), int(m[i, 1]))
Confronti.to_csv('ConfrontiFaticaZE.csv', index=False)

#train the model with the parameters that seemed optimal
n_estimators = 200
max_depth_duplication = 5

skope_rules = SkopeRules(feature_names=feature_labels, random_state=42, n_estimators=n_estimators,
                               recall_min=0.05, precision_min=1,
                               max_samples=0.7,
                               max_depth_duplication= max_depth_duplication, max_depth = 5)
skope_rules.fit(training_set, y_trainNF)

# Visualization of the rules
for i_rule, rule in enumerate(skope_rules.rules_):
    print(rule[0])

#Try out the selected rules by perturbing two features.
#Create a matrix which is the vector product of the perturbations of the two features.
#Calculate the precision and recall parameters for each pair of perturbations and save everything to .csv in order to choose the perturbation that best approximates zero error.
columns = ['Dimensione_dataset', 'Totale_affaticati', 'Totale_non_affaticati', 'Fatigue_previsto',
           'delta1', 'delta2', 'TP', 'FP', 'TN', 'FN', 'error', 'coverage', 'Precision', 'Recall', 'FNR', 'TNR', 'FScore']
Confronti = pd.DataFrame(columns=columns)

brmin=-44.282115151279
brmax=31.7835009073259
br_range=np.arange(brmin,brmax,1)
wjmin=-84.1542088900458
wjmax=39.914009726912
wj_range=np.arange(wjmin, wjmax,1)

y = br_range
x = wj_range
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

Statistiche (MMH_test_set, 0, 0)
for i in range(len(m)):
  Statistiche (MMH_test_set, m[i, 1], m[i, 0])
Confronti.to_csv('ConfrontiNE.csv', index=False)
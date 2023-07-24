
# path to dataset folder
data_dir = '/Users/saranarteni/Documents/Synology_drive_Sara/SynologyDrive/codiceSafety/Dati/'
# training set 
datafilename = data_dir + "platooning_test.xlsx"
#test set
testfile = data_dir + "platooning_test.xlsx"
# name of the output column
class_label = "collision"
# set to True in case of FPR minimization, otherwise set to False and the method will minimize FNR
minimizeFPR = False
# set to True to print the results for each candidate threshold
save_res = False

# specify the method: either "outside" or "inside"
method = "outside"

# list of value ranking intervals
vrstringlist = ["PER > 0.43", "F0 <= -7.50"]

# set the steps for defining the granularity of thresholds tuning (list with 1 value per interval in vrstringlist)
steps = [0.05, 0.5]
assert(len(steps)==len(vrstringlist))


import sys
import pandas as pd
import time

from config import *

from parseVR import *
from load_dataset import *
from getDeltaMatrix import *
from main_algorithm import *
from SafetyRegionSearch import *
from safetyregionEval import *


pd.set_option('display.max_rows', None, 'display.max_columns', None)


# MAIN

SafetyFromValueRanking(vrstringlist, method, datafilename, class_label,steps,testfile, save_res, minimizeFPR)
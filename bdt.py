import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import csv
import pickle

import xgboost as xgb
from xgboost import plot_importance

import utils as utils
from utils import set_font

use_font_size=18
use_legend_font_size=16
use_legtit_font_size=16
plt.rcParams.update({'font.size': use_font_size})
#plt.rcParams.update({'font.family': 'sans'})
set_font()

#-----------------------------------------------------------------------
# main:
#-----------------------------------------------------------------------
verbose = True
# cut sample (speed in code dev)
small_sample = False
# input diagnostic plots:
plot_inputs = False

# read in signal an bkg train data 
sig = pd.read_csv('inputs/sig.csv')
bkg = pd.read_csv('inputs/bkg.csv')
# read in ti-data, to make predictions on the fly
ti_data = pd.read_csv('inputs/ti_data.csv')

cols = ['y1_pt', 'y1_eta', 'y1_phi', 'y1_e','y2_pt', 'y2_eta', 'y2_phi', 'y2_e', 'myy']

sig.columns = cols
bkg.columns = cols
ti_data.columns = cols

#ensure sig/bkg label is 1/0
sig['label']=1
bkg['label']=0
# dummy label for TI data, for uniform format
ti_data['label']=-1

#print(sig.head(5))
#print(bkg.head(5))

# concatenate sig & background, randomize
data = pd.concat([sig,bkg], ignore_index=True)

# randomize rows such that dataframe has uniform properties
data = data.sample(frac=1).reset_index(drop=True)
nev=data.shape[0]
testfrac=0.5
ntest=int(round(nev*testfrac))

# set label
Y = data.label
X = data.drop(['label'], axis=1)

# manual scaling: 
X['y1_pt']=X['y1_pt']/X['myy']
X['y2_pt']=X['y2_pt']/X['myy']
X['y1_e']=X['y1_e']/X['myy']
X['y2_e']=X['y2_e']/X['myy']

myy = data.myy
X = X.drop(['myy'], axis=1)

#todrop = ['myy']
#X = X.drop(['myy'], axis=1)
#for vdrop in todrop:
#    X = X.drop([vdrop], axis=1)
#    ti_X = X.drop([vdrop], axis=1)

# process TI data:
ti_Y = ti_data.label
ti_X = ti_data.drop(['label'], axis=1)

ti_X['y1_pt']=ti_X['y1_pt']/ti_X['myy']
ti_X['y2_pt']=ti_X['y2_pt']/ti_X['myy']
ti_X['y1_e']=ti_X['y1_e']/ti_X['myy']
ti_X['y2_e']=ti_X['y2_e']/ti_X['myy']

ti_myy = ti_data.myy
ti_X = ti_X.drop(['myy'], axis=1)

# prepare test, train, predict objects 
dtest = xgb.DMatrix(X[0:ntest], label=Y[0:ntest])
dtrain = xgb.DMatrix(X[ntest:nev], label=Y[ntest:nev])
dti = xgb.DMatrix(ti_X, label=ti_Y)

# https://xgboost.readthedocs.io/en/latest/parameter.html
param = {'max_depth':8, 'eta':0.3, 'objective':'binary:logistic'}
#param = {'max_depth':6, 'eta':0.1, 'objective':'binary:logitraw'}
num_round = 200
bst = xgb.train(param, dtrain, num_round)

plot_importance(bst)
plot_importance(bst,importance_type="gain")
plt.savefig("features.png",bbox_inches='tight', dpi=1000)
plt.show()

# importance dict:
##importance=bst.get_fscore(importance_type="gain").items()
##f = open("importance.pkl","wb")
## dict-items returned by items() not pickable => convert to list
##pickle.dump(list(importance),f)
##f.close()
##w = csv.writer(open("importance.csv", "w"))
##for key, val in importance:
##    w.writerow([key, val])

#pickle.dumps(list(d.items()))

# make prediction
preds = bst.predict(dtest)
pred = pd.DataFrame(data=preds, index=range(0,preds.size), columns=['pred'])
#pX = pd.DataFrame(data=X[0:ntest], index=range(0,preds.size))
result = pd.concat([Y[0:ntest],X[0:ntest],myy[0:ntest],pred], axis=1)

print('storing prediction results')
print(result.head(10))
print(result.shape)
result.to_csv('inputs/bdt_out.csv')

print('running predictions on TI data:')
ti_preds = bst.predict(dti)
ti_pred = pd.DataFrame(data=ti_preds, index=range(0,ti_preds.size), columns=['pred'])
ti_result = pd.concat([ti_Y,ti_X,ti_myy,ti_pred], axis=1)
print('storing prediction results')
print(ti_result.head(10))
print(ti_result.shape)
ti_result.to_csv('inputs/ti_bdt_out.csv')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import roc_curve, auc

import utils as utils
from utils import set_font

use_font_size=18
use_legend_font_size=16
use_legtit_font_size=16
plt.rcParams.update({'font.size': use_font_size})
#plt.rcParams.update({'font.family': 'sans'})
set_font()

data=pd.read_csv('bdt_out.csv')
Y = data['label'].values
PRED = data['pred'].values
ti_data=pd.read_csv('ti_bdt_out.csv')

fpr, tpr, thresholds = roc_curve(Y, PRED, pos_label=1)
roc_auc = auc(fpr,tpr)

plt.figure()
lw=1
plt.plot(fpr, tpr, color='orange',
         lw=lw, label='Area U.C. = %0.2f' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate',horizontalalignment='right', x=1.0)
plt.ylabel('True Positive Rate',horizontalalignment='right', y=1.0)
plt.savefig("roc.png", bbox_inches='tight')
plt.savefig("roc.pdf", bbox_inches='tight')
plt.show()

# discriminant 
plt.figure()
bins = np.linspace(0,1,51)
_sig=data[data['label']==1]
_bkg=data[data['label']==0]
values,_,__=plt.hist(_sig['pred'], bins, color='firebrick',
                     histtype='step',
                     label=r'gg$\rightarrow$H$\rightarrow \gamma\gamma$',
                     weights=utils.get_w(_sig['pred']))

values,_,__=plt.hist(_bkg['pred'], bins, color='green',linestyle='--',
                     histtype='step',
                     label=r'background',
                     weights=utils.get_w(_bkg['pred']))

values,_,__=plt.hist(ti_data['pred'], bins, color='black',linestyle='--',
                     histtype='step',
                     label=r'background',
                     weights=utils.get_w(ti_data['pred']))

plt.xlabel('ML discriminant',horizontalalignment='right', x=1.0)
plt.ylabel('Fraction of events/0.02',horizontalalignment='right', y=1.0)

yminrange=0.0
ymaxrange=0.1311
yspan=ymaxrange-yminrange
xminrange=0.0
xmaxrange=1.0
xrange=xmaxrange-xminrange
plt.xlim([0.0, xmaxrange])
plt.ylim([yminrange, ymaxrange])

#plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right",prop={'size': use_legend_font_size},frameon=False)
#plt.text(0.015, 0.008,"Sherpa 2.2.0 @ NLO\nPhoton Selection:\n"+r'$\mathrm{p}_\mathrm{T}(\gamma1) \cdot \mathrm{p}_\mathrm{T}(\gamma2)>20\,\mathrm{GeV}^2$'+ "\nATLAS detector response:\nRivet folding",fontsize=use_legtit_font_size, color='gray',verticalalignment="bottom")
#plt.text(0.015, 0.008,"Sherpa 2.2.0 @ NLO"+"\nATLAS detector response:\nRivet folding",fontsize=use_legtit_font_size, color='gray',verticalalignment="bottom")
#plt.text(0.015, ymaxrange-0.12*yspan,"CHAOS",fontsize=use_font_size, verticalalignment="bottom", color='black')

plt.savefig("bdt_disc.png", bbox_inches='tight', dpi=500)
plt.savefig("bdt_disc.pdf", bbox_inches='tight')
plt.show()  

# finally, myy plot for events passing the cut
plt.figure()
pred_cut=0.85

myy_low=105.
myy_high=160.
myy_sig_low=121.
myy_sig_high=129.

data=(data[data['pred']>pred_cut])
data=data[(data['myy']>myy_low) & (data['myy']<myy_high)]

ti_data=ti_data[ti_data['pred']>pred_cut]
ti_data=ti_data[(ti_data['myy']>myy_low) & (ti_data['myy']<myy_high)]

_sig=data[data['label']==1]
_bkg=data[data['label']==0]

nsig_sigreg = _sig[(_sig['myy']>myy_sig_low) & (_sig['myy']<myy_sig_high)].shape[0]
nbkg_sigreg= _bkg[(_bkg['myy']>myy_sig_low) & (_bkg['myy']<myy_sig_high)].shape[0]
nti_sigreg = ti_data[(ti_data['myy']>myy_sig_low) & (ti_data['myy']<myy_sig_high)].shape[0]
#nbkg_bkgreg= _bkg[(_bkg['myy']<myy_sig_low)].shape[0]+ _bkg[(_bkg['myy']>myy_sig_high)].shape[0]
nsig = _sig.shape[0]
nbkg = _bkg.shape[0]
nti=ti_data.shape[0]
print(nti,nti_sigreg)

bins = np.linspace(myy_low,myy_high,int(myy_high-myy_low))
#values,_,__=plt.hist(_sig['myy'], bins, color='firebrick',
 #                    histtype='step',
 #                    label=r'$H$\rightarrow \gamma\gamma$ signal')
#                     weights=utils.get_w(_sig['pred']))
#sig_int=utils.get_integral(values,bins)
#sig_int_sigrreg=utils.get_integral(values,bins,121,129)

values,_,__=plt.hist(_bkg['myy'], bins, color='green',linestyle='--',
                     histtype='step',
                     label=r'background')
#                     weights=utils.get_w(_bkg['pred']))
#bkg_int=utils.get_integral(values,bins)
#bkg_int_sigreg = utils.get_integral(values,bins,121,129)
#bkg_int_bkgreg = bkg_int - bkg_int_sigreg

values,_,__=plt.hist(ti_data['myy'], bins, color='black',linestyle='--',
                     histtype='step',
                     label=r'background')
#                     weights=utils.get_w(ti_data['pred']))
#ti_int = utils.get_integral(values,bins)
#ti_int_sigreg = utils.get_integral(values,bins,121,129)
#ti_int_bkgreg = ti_int-ti_int_sigreg

plt.xlabel(r'm$_{\gamma\gamma}$ [GeV]',horizontalalignment='right', x=1.0)
plt.ylabel('Events/1 GeV',horizontalalignment='right', y=1.0)

plt.xlim([myy_low,myy_high])
plt.savefig("myy.png", bbox_inches='tight', dpi=500)
plt.savefig("myy.pdf", bbox_inches='tight')
plt.show()

#print("raw number of signl events & evenst under mass peak:", sig_int,sig_int_sigrreg)
#print("raw number of bkg events & evenst under mass peak:", bkg_int,bkg_int_sigreg)
#print("raw number of TI data events ", ti_int)

exit(0)

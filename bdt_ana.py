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

#-----------------------------------------------------------------------

_sig=data[data['label']==1]
_bkg=data[data['label']==0]

#-----------------------------------------------------------------------

# discriminant 
plt.figure()

myy_sig_low = 121
myy_sig_high = 129
ti_data_sb_low=ti_data[(ti_data['myy']<myy_sig_low)]
ti_data_sb_high=ti_data[(ti_data['myy']>myy_sig_high)]
ti_data_sb = pd.concat([ti_data_sb_low,ti_data_sb_high], ignore_index=True)

bkg_sb_low=_bkg[(_bkg['myy']<myy_sig_low)]
bkg_sb_high=_bkg[(_bkg['myy']>myy_sig_high)]
bkg_sb = pd.concat([bkg_sb_low,bkg_sb_high], ignore_index=True)

bins = np.linspace(0,1,51)

values_ti_data,edges,__=plt.hist(ti_data_sb['pred'], bins, color='green',linestyle='--',
                                 histtype='step',
                                 label=r'Data',
                                 weights=utils.get_w(ti_data_sb['pred']))
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.clf()

#sig:
values,_,__=plt.hist(_sig['pred'], bins, color='firebrick',
                     histtype='step',
                     linewidth=2,
                     label=r'H$\rightarrow \gamma\gamma$ signal',
                     weights=utils.get_w(_sig['pred']))
# background:
values,_,__=plt.hist(bkg_sb['pred'], bins, color='green',
                     linestyle='--',
                     histtype='step',
                     linewidth=2,
                     label=r'Background side-band',
                     weights=utils.get_w(bkg_sb['pred']))
# TI data
n_ti_data=ti_data_sb.shape[0]
plt.errorbar(bin_centers, values_ti_data, yerr=np.sqrt(1/n_ti_data)*np.sqrt(values_ti_data),fmt='.k',label='Data', markersize=8)

###################
##values,_,__=plt.hist(_sig['pred'], bins, color='firebrick',
##                     histtype='step',
##                     label=r'gg$\rightarrow$H$\rightarrow \gamma\gamma$',
##                     weights=utils.get_w(_sig['pred']))
##
##values,_,__=plt.hist(_bkg['pred'], bins
##                     , color='green',linestyle='--',
##                     histtype='step',
##                     label=r'background',
##                     weights=utils.get_w(_bkg['pred']))
##
##values,_,__=plt.hist(ti_data['pred'], bins, color='black',linestyle='--',
##                     histtype='step',
##                     label=r'background',
##                     weights=utils.get_w(ti_data['pred']))
##

# ------ axis & text settings  -----------------------------

xminrange=0.0
xmaxrange=1.0
xspan=xmaxrange-xminrange

plt.xlim([0.0, xmaxrange])

# ------------------------------ lin scale -----------------------------
yminrange=0.0
ymaxrange=0.1311
yspan=ymaxrange-yminrange
stamp_x = xminrange+0.015*xspan
stamp_y = ymaxrange-0.1*yspan
lum_x = xminrange+0.35*xspan
lum_y = ymaxrange-0.45*yspan
epsilon_xtext = 0.01
epsilon_ytext = 0.01
plt.ylabel('Fraction of events/0.02',horizontalalignment='right', y=1.0)
# ------------------------------ log scale -----------------------------
dolog = True
if (dolog):
    plt.yscale('log')
    yminrange=0.0001
    ymaxrange=0.9
    yspan=np.log10(ymaxrange)-np.log10(yminrange)
    print('yspan is:',yspan)
    stamp_y = 0.5
    lum_y = 0.05
    epsilon_ytext = 0.

# ---------------------------------------- -----------------------------
plt.ylim([yminrange, ymaxrange])
plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right",prop={'size': use_legend_font_size},frameon=False)

plt.text(stamp_x,stamp_y ,"CHAOS",fontsize=use_font_size, verticalalignment="bottom", color='black')
plt.text(lum_x, lum_y,r'$\int$L=10 fb$^{-1}$,$\sqrt{s}$=13 TeV',fontsize=use_font_size, verticalalignment="bottom", color='black')
plt.text(xmaxrange+epsilon_xtext*xspan,yminrange+epsilon_ytext*yspan, "CERN Open Data + FindTheHiggs", fontsize=14, color='gray',verticalalignment="bottom",rotation="vertical")

plt.xlabel('ML discriminant',horizontalalignment='right', x=1.0)


plt.savefig("bdt_disc.png", bbox_inches='tight', dpi=500)
plt.savefig("bdt_disc.pdf", bbox_inches='tight')
plt.show()  

#-----------------------------------------------------------------------
# S and B normalization 
# TODO: set
pred_cut=0.85

myy_low=105.
myy_high=160.
myy_sig_low=121.
myy_sig_high=129

data=(data[data['pred']>pred_cut])
data=data[(data['myy']>myy_low) & (data['myy']<myy_high)]

ti_data=ti_data[ti_data['pred']>pred_cut]
ti_data=ti_data[(ti_data['myy']>myy_low) & (ti_data['myy']<myy_high)]

nsig_sigreg = _sig[(_sig['myy']>myy_sig_low) & (_sig['myy']<myy_sig_high)].shape[0]
nbkg_sigreg= _bkg[(_bkg['myy']>myy_sig_low) & (_bkg['myy']<myy_sig_high)].shape[0]
nti_sigreg = ti_data[(ti_data['myy']>myy_sig_low) & (ti_data['myy']<myy_sig_high)].shape[0]
#nbkg_bkgreg= _bkg[(_bkg['myy']<myy_sig_low)].shape[0]+ _bkg[(_bkg['myy']>myy_sig_high)].shape[0]
nsig = _sig.shape[0]
nbkg = _bkg.shape[0]
nti=ti_data.shape[0]
print("ntni data: total, sig region:", nti,nti_sigreg)

# set norm factors for sig and BKG predictions, based on TI data:
expected_sigreg_yield = 120. # to be calculated from theo prediction 
expected_bkgreg_yield = (nti-nti_sigreg)*(1.+nbkg_sigreg/nbkg)

bins = np.linspace(myy_low,myy_high,int(myy_high-myy_low))
#sig_int=utils.get_integral(values,bins)
#print("signal yield", sig_int)
#sig_int_sigrreg=utils.get_integral(values,bins,121,129)

#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# finally, myy plot for events passing the cut
plt.figure()

# plot boundary settings: 
yminrange=0.0
ymaxrange=38.0
yspan=ymaxrange-yminrange
xminrange=myy_low
xmaxrange=myy_high
xspan=xmaxrange-xminrange


##### v1: plot w/o TI data: 
# 
##values_bkg,edges,__=plt.hist(_bkg['myy'], bins, color='green',linestyle='--',
##                     histtype='step',
##                     label=r'background',
##                     weights=expected_bkgreg_yield*utils.get_w(_bkg['pred']))
##bin_centers = 0.5 * (edges[:-1] + edges[1:])
##
##plt.clf()
##
### background:
##plt.errorbar(bin_centers, values_bkg, yerr=np.sqrt(values_bkg),fmt='.k',label='Expected background')
###sig:
##values,_,__=plt.hist(_sig['myy'], bins, color='firebrick',
##                     histtype='step',
##                     linewidth=2,
##                     label=r'H$\rightarrow \gamma\gamma$ signal',
##                     weights=expected_sigreg_yield*utils.get_w(_sig['pred']))
##
## plt.text(xminrange+0.35*xspan, ymaxrange-0.35*yspan,r'$\int$L=10 fb$^{-1}$,$\sqrt{s}$=13 TeV',fontsize=use_font_size, verticalalignment="bottom", color='black')

##### v2: plot with TI data:
yminrange=0.0
ymaxrange=54.0
yspan=ymaxrange-yminrange

values_ti_data,edges,__=plt.hist(ti_data['myy'], bins, color='green',linestyle='--',
                     histtype='step',
                     label=r'Data')
bin_centers = 0.5 * (edges[:-1] + edges[1:])

plt.clf()

#sig:
values,_,__=plt.hist(_sig['myy'], bins, color='firebrick',
                     histtype='step',
                     linewidth=2,
                     label=r'H$\rightarrow \gamma\gamma$ signal',
                     weights=expected_sigreg_yield*utils.get_w(_sig['pred']))
# background:
values,_,__=plt.hist(_bkg['myy'], bins, color='green',
                     linestyle='--',
                     histtype='step',
                     linewidth=2,
                     label=r'Estimated background',
                     weights=expected_bkgreg_yield*utils.get_w(_bkg['pred']))
# TI data
plt.errorbar(bin_centers, values_ti_data, yerr=np.sqrt(values_ti_data),fmt='.k',label='Data', markersize=8)
plt.text(xminrange+0.35*xspan, ymaxrange-0.45*yspan,r'$\int$L=10 fb$^{-1}$,$\sqrt{s}$=13 TeV',fontsize=use_font_size, verticalalignment="bottom", color='black')
#------------------------------------------------------------------------------------------------

plt.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right",prop={'size': use_legend_font_size},frameon=False)
plt.text(xminrange+0.015*xspan, ymaxrange-0.1*yspan,"CHAOS",fontsize=use_font_size, verticalalignment="bottom", color='black')
plt.text(xmaxrange+0.01*xspan,yminrange+0.01*xspan, "CERN Open Data + FindTheHiggs", fontsize=14, color='gray',verticalalignment="bottom",rotation="vertical")

plt.xlabel(r'm$_{\gamma\gamma}$ [GeV]',horizontalalignment='right', x=1.0)
plt.ylabel('Events/1 GeV',horizontalalignment='right', y=1.0)

plt.ylim([yminrange,ymaxrange])
plt.xlim([xminrange,xmaxrange])
plt.savefig("myy.png", bbox_inches='tight', dpi=500)
plt.savefig("myy.pdf", bbox_inches='tight')
plt.show()

#print("raw number of signl events & evenst under mass peak:", sig_int,sig_int_sigrreg)
#print("raw number of bkg events & evenst under mass peak:", bkg_int,bkg_int_sigreg)
#print("raw number of TI data events ", ti_int)

exit(0)

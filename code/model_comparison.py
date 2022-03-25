
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import arviz as az
from bambi import Prior
from scipy.stats import halfnorm
from scipy.stats.kde import gaussian_kde
from pathlib import Path
sns.set_context("talk")

folder='E:/ALBERTO/expe20-21/repetition_def/psych_sci_commentary/psych_sci_commentary/'
fold=Path(folder)
datafold='data/'
datafolder = Path(datafold)
file = 'dataT_T.csv'
filename = folder / datafolder / file
data=pd.read_csv(filename)#read datafile

#%% CALCULATE PROBABILITY FIRST SEEN AS REPETITION 


long=pd.melt(data, id_vars=['suje','DISTR'],
             value_vars=['2','3','4','5','6','7','8','9','10'],
        var_name='REPE', value_name='acc')#long format
long['acc']=long['acc']/100# 0 to 1 values
long[['REPE','DISTR']]=long[['REPE','DISTR']].astype(int)
long.reset_index(inplace=True)
long=long.sort_values(['suje','DISTR','REPE'],axis=0)

# p-first= ((accuracy at repetition x) - (accuracy at repetition x-1)) /(1-accuracy at repetition x-1)
long['p_first']=(long.acc-long.acc.shift(1))/(1-long.acc.shift(1))
long['p_first']=np.where(long['REPE']==2,long['acc'],long['p_first'])#p_first at Repetition 2 = accuracy at Repetition 2
long=long.replace([np.inf, -np.inf], np.nan)




#%% ONE DISTRACTOR, MODEL COMPARISON 

one_dis=long[(long['DISTR']==1)&(long['REPE']<5)]
one_dis['suje']=one_dis['suje'].astype('category')
#Full model
halfNormalPrior = Prior('HalfNormal',sigma=0.1)
prior_full = {"REPE":halfNormalPrior,"1|suje":"narrow"}
formula="p_first ~ REPE + (1|suje)"
full_model = bmb.Model(data=one_dis,formula=formula,priors=prior_full)
full_fitted = full_model.fit(cores=1, draws=3000,chains=1)
#BAYES FACTOR
posterior_kde = gaussian_kde(full_fitted.posterior['REPE'][:])
posterior_0 = posterior_kde(0) 
BF01 = halfnorm.pdf(0,scale=0.1)/posterior_0
print (f'the Bayes Factor is {BF01}')
#Null model
prior_null = {"1|suje":"narrow"}
formula_null="p_first ~ (1|suje)"
null_model = bmb.Model(data=one_dis,formula=formula_null,priors=prior_null)
null_fitted = null_model.fit(cores=1, draws=3000,chains=1)
#Model comparison
models_dict1 = {
    "full": full_fitted,
    "null": null_fitted}
df_compare = az.compare(models_dict1,scale='deviance')
az.plot_compare(df_compare, insample_dev=True);




#%% TWO DISTRACTROS MODEL COMPARISON

two_dis=long[(long['DISTR']==2)&(long['REPE']<7)]
two_dis['suje']=two_dis['suje'].astype('category')
#FULL MODEL
halfNormalPrior = Prior('HalfNormal',sigma=0.1)
prior_full = {"REPE":halfNormalPrior,"1|suje":"narrow"}
formula="p_first ~ REPE + (1|suje)"
full_model = bmb.Model(data=two_dis,formula=formula,priors=prior_full)
full_fitted = full_model.fit(cores=1, draws=3000,chains=1)
#BAYES FACTOR
posterior_kde = gaussian_kde(full_fitted.posterior['REPE'][:])
posterior_0 = posterior_kde(0)
BF01 = posterior_0/halfnorm.pdf(0,scale=0.1)
print (f'the Bayes Factor is {BF01}')
#Null model
prior_null = {"1|suje":"narrow"}
formula_null="p_first ~ (1|suje)"
null_model = bmb.Model(data=two_dis,formula=formula_null,priors=prior_null)
null_fitted = null_model.fit(cores=1, draws=3000,chains=1)
#Model comparison
models_dict2 = {
    "full": full_fitted,
    "null": null_fitted}
df_compare2 = az.compare(models_dict2,scale='deviance')
az.plot_compare(df_compare2, insample_dev=True);



#%%Plotting deviance for 1 and 2 distractors
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                             gridspec_kw={'hspace':1})
az.plot_compare(df_compare, insample_dev=True,order_by_rank=False,ax=ax1);
az.plot_compare(df_compare2, insample_dev=True,ax=ax2);
ax1.set_title('One distractor')
ax2.set_title('Two distractors')
f.set_size_inches(7,4)
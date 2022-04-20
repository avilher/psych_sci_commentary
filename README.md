# psych_sci_commentary
<div id="top"></div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project



Code to reproduce the plots and results of "No subliminal memory for spaced repeated images in rapid-serial-visual-presentation streams" `avilher`, `avilher/psych_sci_commentary`, `avilher@gmail.com`

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python](https://www.python.org/)
* [Bambi](https://bambinos.github.io/bambi/main/index.html)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Arviz](https://arviz-devs.github.io/arviz/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code requires a working Python interpreter (3.7+). We recommend installing Python and its key numerical libraries using the Anaconda Distribution.

Bambi can be installed PyPI:

* Bambi
  ```sh
  pip install bambi
  ```
 Seaborn can be installed from PyPI:
* Seaborn
  ```sh
  pip install seaborn
  ```


<!-- USAGE EXAMPLES -->
## Usage

* CALCULATE PROBABILITY FIRST SEEN AS REPETITION 
  ```sh
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import matplotlib.patches as mpatches
  from pathlib import Path

  sns.set_context("talk")
  
  #long format
  long = pd.melt(data, id_vars=['suje','DISTR'],
             value_vars=['2','3','4','5','6','7','8','9','10'],
        var_name='REPE', value_name='acc')#long format
  long['acc'] = long['acc']/100# 0 to 1 values
  long[['REPE','DISTR']] = long[['REPE','DISTR']].astype(int)
  long.reset_index(inplace=True)
  long = long.sort_values(['suje','DISTR','REPE'],axis=0)

  # p-first= ((accuracy at repetition x) - (accuracy at repetition x-1)) /(1-accuracy at repetition x-1)
  long['p_first'] = (long.acc-long.acc.shift(1)) / (1-long.acc.shift(1))
  long['p_first'] = np.where(long['REPE']==2,long['acc'],long['p_first'])#p_first at Repetition 2 = accuracy at Repetition 2
  long = long.replace([np.inf, -np.inf], np.nan)
  ```
* FIGURE 
  ```sh
  single = long[(long['DISTR']==1)&(long['REPE']<5)]
  double = long[(long['DISTR']==2)&(long['REPE']<7)]
  fig,ax = plt.subplots()
  g = sns.pointplot(x = 'REPE', y = "p_first", ci = 68, color = 'slateblue',
                  ax = ax, data = single)
  sns.pointplot(x = 'REPE', y = "p_first", color = 'r', linestyles = '--',
              ci=68, ax = ax, data = double)
  sns.despine()
  g.set(ylim = (0,1), xlim = (-0.5,4.2),
      xlabel = 'Number of Presentations',
      ylabel = 'Probability')
  red_patch = mpatches.Patch(color='r', label='Two')
  blue_patch = mpatches.Patch(color='slateblue', label='One')
  plt.legend(handles = [blue_patch,red_patch],
            loc = 'upper right', 
           frameon = False, title = 'No. of distractors')
  fig.set_size_inches(7,6)
  ```
* ONE DISTRACTOR MODEL COMPARISON
  ```sh
  one_dis = long[(long['DISTR'] == 1) & (long['REPE'] < 5)]
  one_dis['suje'] = one_dis['suje'].astype('category')
  #Full model
  halfNormalPrior = Prior('HalfNormal', sigma = 0.1)
  prior_full = {"REPE" : halfNormalPrior, "1|suje":"narrow"}
  formula = "p_first ~ REPE + (1|suje)"
  full_model = bmb.Model(data = one_dis, formula = formula, priors = prior_full)
  full_fitted = full_model.fit(cores = 1, draws = 3000, chains = 1)
  #BAYES FACTOR for the Alternative
  posterior_kde = gaussian_kde(full_fitted.posterior['REPE'][:])
  posterior_0 = posterior_kde(0) 
  BF10 = halfnorm.pdf(0,scale = 0.1) / posterior_0
  print (f'the Bayes Factor is {BF10}')
  #Null model
  prior_null = {"1|suje":"narrow"}
  formula_null = "p_first ~ (1|suje)"
  null_model = bmb.Model(data = one_dis, 
                       formula = formula_null, priors = prior_null)
  null_fitted = null_model.fit(cores = 1, draws = 3000, chains = 1)
  #Model comparison
  models_dict1 = {"full" : full_fitted, "null" : null_fitted}
  df_compare = az.compare(models_dict1, scale = 'deviance')
  az.plot_compare(df_compare, insample_dev = True);
  ```
* TWO DISTRACTORS MODEL COMPARISON
  ```sh
  two_dis = long[(long['DISTR'] == 2)&(long['REPE'] < 7)]
  two_dis['suje'] = two_dis['suje'].astype('category')
  #FULL MODEL
  halfNormalPrior = Prior('HalfNormal', sigma = 0.1)
  prior_full = {"REPE" : halfNormalPrior, "1|suje":"narrow"}
  formula = "p_first ~ REPE + (1|suje)"
  full_model = bmb.Model(data = two_dis, formula = formula, priors = prior_full)
  full_fitted = full_model.fit(cores = 1, draws = 3000, chains = 1)
  #BAYES FACTOR
  posterior_kde = gaussian_kde(full_fitted.posterior['REPE'][:])
  posterior_0 = posterior_kde(0)
  BF01 = posterior_0 / halfnorm.pdf(0, scale = 0.1)
  print (f'the Bayes Factor is {BF01}')
  #Null model
  prior_null = {"1|suje":"narrow"}
  formula_null = "p_first ~ (1|suje)"
  null_model = bmb.Model(data = two_dis,
                       formula = formula_null, priors = prior_null)
  null_fitted = null_model.fit(cores = 1, draws = 3000, chains = 1)
  #Model comparison
  models_dict2 = {"full" : full_fitted, "null" : null_fitted}
  df_compare2 = az.compare(models_dict2, scale = 'deviance')
  az.plot_compare(df_compare2, insample_dev = True);
  ```
  
<p align="right">(<a href="#top">back to top</a>)</p>






<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alberto Aviles - [@generalife22](https://twitter.com/generalife22) - avilher@gmail.com
Project Link: [https://github.com/avilher/psych_sci_commentary](https://github.com/avilher/psych_sci_commentary)

<p align="right">(<a href="#top">back to top</a>)</p>


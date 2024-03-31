# CO2_recursive_timeseries
Recursive multi-step Time series forcasting for CO2 level


## Dataset
Here is part of the dataset:

| time | co2 |
| --- | --- |
| 3/29/1958 | 316.1 |
| 4/5/1958 | 317.3 |
| 4/12/1958 |	317.6 |
| 4/19/1958 |	317.5 |
| 4/26/1958 |	316.4 |
| 5/3/1958 | 316.9 |
| 5/10/1958	| |
| 5/17/1958 | 317.5 |
| 5/24/1958 | 317.9 |
| 5/31/1958 | |
| 6/7/1958 | |
| 6/14/1958	| |
| 6/21/1958	| |
| 6/28/1958	| |
| 7/5/1958 | 315.8 |
| 7/12/1958 | 315.8 |
| 7/19/1958 |	315.4 |
| 7/26/1958 |	315.5 |


## Pre-processing
* Convert the date in the right format <br />
* Complete the missing values <br />
<br />


## Training
Use of 5 last weeks to predict the 6th week. <rb />
<br />

I choose the Linear Regression model because a big correlation between features. <br />
<br />

Performances: <br />
R2: 0.9907505918201436 <br />
MSE: 0.22044947360346612 <br />
MAE: 0.3605603788359238 <br />
<br />

The Linear Regression model is quite good! <br />
<br />


Let's visualize the prediction: <br />
![image](https://github.com/T-KIEU/CO2_recursive_timeseries/assets/100022674/46788a97-89bd-4dd0-8085-f16176008183)



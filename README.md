# Mortgage Disparate Treatment Fair Lending Analysis

Test for potential areas of risk and includes Propensity Score Analysis as a way to estimate bias in estimates.




# Project overview



I use application level data to shows how fair lending disparate treatment analysis is used to identify potential areas of risk for mortgages. The purpose of disparate treatment analysis in fair lending is to determine whether similarly situated applicants or borrowers are treated differently because of a protected characteristic, such as race, ethnicity, sex, etc. This project includes segmenting, picking protected basis groups for tests, and estimating with and without controlling for consumer and loan attributes. The project shows that incorporating Propensity Score Analysis helps in identifying potential bias in estimating effects of disparate treatment and can be used to make more conservative decisions.



**Disclaimer:** the project does not aim to identify actual risk rather it 1) presents the testing framework and 2) proposes value add of incorporating Propensity Score Analysis. In practice internal data would be used together with knowledge of underwriting and pricing strategies to estimate effects. While there are industry practices on criteria for identifying areas of risk, these are not used since that is not the intent of this project.




# Problem Statement


Lenders are prohibited from making lending decisions based on protected basis. As part of adhering to regulatory requirements, lenders conduct *file reviews*. These are detailed examination of individual loan files to evaluate whether similarly situated applicants or borrowers were treated consistently and in compliance with fair lending requirements. However, given the volume of applications it is not possible or cost effective to compare every individual loan. In practice, institutions leverage regression analysis to do hypothesis tests on segments for each protected basis group. Generally, an effect must be larger than a predetermined magnitude and significant to identify potential risk. This regression analysis identifies *potential areas of risk* which are then looked at in more detail during a targeted *file review*.


The project aims to answer the following two questions.

- How is fair lending disparate treatment regression analysis conducted?
- Is there bias in estimated effects?




# Dataset


I use Modified Loan Application Register (MLAR) data from the Consumer Financial Protection Bureau (CFPB) that is made public as part of the Home Mortgage Disclosure Act (HMDA). Analysis uses all loans from 2025 for a single nationwide financial institution. 



Data originally contains 368,994 rows and 85 columns. Target values are 'interest rate' for pricing analysis and 'action taken' for underwriting analysis. Action taken identifies denied, booked, and unboked, which is the population of underwriting analysis. Pricing uses booked and unbooked since they both receive interest rates in the data.



The dataset contains 7,043 customer records and 21 features.

- **Target:** `interest rate` for pricing, `action taken` for underwriting
- **Features:** demographics, consumer credit attributes, loan attributes
- **Source:** Consumer Financial Protection Bureau (CFPB)




# Tools and Technologies

- Python
- pandas
- statsmodels
- Matplotlib
- Seaborn
- Toad
- Jupyter Notebook


## Project Structure


```text
hmda-disp-treat/
├── code/
│   ├── utils/ (contains functions used in notebooks)
│   │   ├── __init__.py
│   │   ├── describe.py
│   │   ├── model.py
│   │   └── results.py
│   ├── 00 get data.ipynb
│   ├── 01 prep data.ipynb
│   ├── 02 descriptive, segments, and tests.ipynb
│   ├── 03 remove columns we dont use.ipynb
│   ├── 04a models race.ipynb
│   ├── 04b models gender.ipynb
│   ├── 04c models age.ipynb
│   ├── 05 results.ipynb
│   └── requirements.txt/
├── data/ (00 notebook pulls data from cfpb website, data is publicly available)
├── README.md
└── .gitignore
```




# Methodology

Notebooks 00 to 03 filter data to a more narrow definition of what our tests will be. This includes adding descriptions from data dictionary, removing columns that are not usable or leak information the model shouldn't know. It defines segments and the protected basis we will test.

Notebooks 04x run the regression analysis for each of the tests. Below are the protected basis and reference groups the individual regressions test.

Race: Black African American vs White
Gender: Female vs Male
Age: over 62 vs under or equal to 62






The following regressions are ran for both underwriting and pricing data. While not described, the same general equations are estimated for pricing using OLS.






**Model 0**


$$
\Pr(denial_i = 1 \mid PB_i)
=
\beta_0 + \beta_1 PB_i
$$





**Model 1**


$$
\Pr(denial_i = 1 \mid PB_i, \mathbf{X}_i)
=
\beta_0 + \beta_1 PB_i + \mathbf{X}_i^\top\boldsymbol{\beta}
$$




**Model 2**


PSA: first we estimate a propensity score from the probability of *treatment*, in this case observing the PB. Then matching observations across treatment and control groups which have similar propensity scores. Having a similar propensity score means that they have similar covariates, the only difference then is that one is in the PB group and the other in the reference. In essence, we are making the observed PB rate random and it should not be explainable from covariates. This is what gives us the neater interpretation of unbiased treatment effect. 


$$
\Pr(PB_i = 1 \mid \mathbf{X}_i)
=
\beta_0 +  \mathbf{X}_i^\top\boldsymbol{\beta}
$$


The matched pair dataset is then used to run the same equation as in model 1. However, this time the interpretation of the coefficient for PB is assumed to be unbiased.



$$
\Pr(denial_i = 1 \mid PB_i, \mathbf{X}_i)
=
\beta_0 + \beta_1 PB_i + \mathbf{X}_i^\top\boldsymbol{\beta}
$$





# Key Findings

## underwriting

- Cash out refinance FHA insured Female
    - magnitude of effect increases
    - PB coefficient becomes significant
    - model 2 shows higher fit
    - separation shows the model explains approvals and declines better
- Home purchase conventional Yes(62+)
    - magnitude of effect increases
    - PB coefficient remains significant
    - model 2 shows higher fit
    - separation shows model explains approvals and declines better

For both of these tests there is evidence that suggest bias in the respective model 1 regressions. There is likely to have been a different outcome in identifying potential risk when only considering model 1, supporting the value add of incorporating PSA into analysis.



## pricing

- In general effect of PB on interest rate remains similar across model 1 and 2.
- While magnitudes don't change much, significance of effect changed across model 1 and 2 in about half of the tests.
- Model fit across most tests improved across models 1 and 2 with few exceptions.


While magnitude of coefficients doesn't change much identifying potential areas of risk also depends on significance. There is likely to have been a different outcome in identifying potential risk for these test under model 1, supporting the value add of incorporating PSA into analysis.











# 12. Limitations




- Unknown what the underwriting and pricing strategy really is for this (any) lender from public information. Having it would lead to better model specifications.
- Limited data on consumer and loan attributes. One key example is not having consumer credit score, public data only has attributes such as LTV and DTI.
- Simpler definitions of protected basis. Analysis conducted is valid although for a narrower population. More complicated analysis could be done on joint applicants, same sex applicants, mixed race applicants, etc. This would require more work on justifying how to derive these PB groups based on multiple characteristics.






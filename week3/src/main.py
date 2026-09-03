"""
Week 3 — Statistical Analysis and Hypothesis Testing
Question: Is survival independent of passenger sex?

H0: Survival and sex are independent.
H1: Survival and sex are associated.

The canonical 891-row Titanic dataset gives the following sex × survival counts:
Female: 233 survived, 81 did not.
Male:   109 survived, 468 did not.
"""

from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

BASE=Path(__file__).resolve().parents[1]
RAW=BASE/"data/raw/titanic.csv"
FIG=BASE/"outputs/figures"
URL="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
RAW.parent.mkdir(parents=True,exist_ok=True)
FIG.mkdir(parents=True,exist_ok=True)

urllib.request.urlretrieve(URL,RAW)
df=pd.read_csv(RAW)

# Contingency table
table=pd.crosstab(df["sex"],df["survived"])
print(table)

# Pearson chi-square test of independence
chi2,p,dfree,expected=stats.chi2_contingency(table,correction=False)
print("Chi-square:",chi2)
print("p-value:",p)
print("Degrees of freedom:",dfree)

# Fisher's exact test as a robustness check
odds_ratio,fisher_p=stats.fisher_exact(table)
print("Odds ratio:",odds_ratio)
print("Fisher p-value:",fisher_p)

# Two-proportion z-test
counts=np.array([table.loc["female",1],table.loc["male",1]])
totals=np.array([table.loc["female"].sum(),table.loc["male"].sum()])
z,z_p=proportions_ztest(counts,totals)
print("Two-proportion z:",z)
print("Two-proportion p:",z_p)

# Wilson 95% confidence intervals
for sex in ["female","male"]:
    successes=int(table.loc[sex,1])
    total=int(table.loc[sex].sum())
    ci=proportion_confint(successes,total,method="wilson")
    print(sex,"rate:",successes/total,"95% CI:",ci)

# Difference in proportions and 95% CI
p1=counts[0]/totals[0]
p2=counts[1]/totals[1]
diff=p1-p2
se=np.sqrt(p1*(1-p1)/totals[0]+p2*(1-p2)/totals[1])
diff_ci=(diff-1.96*se,diff+1.96*se)
print("Female - male difference:",diff)
print("95% CI:",diff_ci)

# Effect size for a 2x2 table
phi=np.sqrt(chi2/df["survived"].count())
print("Phi effect size:",phi)

# Visualizations
rates=df.groupby("sex")["survived"].mean()
rates.plot(kind="bar",ylim=(0,1),title="Survival Rate by Sex")
plt.ylabel("Survival proportion")
plt.tight_layout()
plt.savefig(FIG/"01_survival_rate_by_sex.png",dpi=220)
plt.close()

print("All tests and figures completed.")

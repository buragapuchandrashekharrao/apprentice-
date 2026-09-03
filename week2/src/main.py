from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data/raw/titanic.csv"
FIG = BASE / "outputs/figures"
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

RAW.parent.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Download exact public dataset
urllib.request.urlretrieve(URL, RAW)
df = pd.read_csv(RAW)

sns.set_theme(style="whitegrid", context="notebook")

# 1. Survival by sex and class
rates = df.pivot_table(index="pclass", columns="sex",
                       values="survived", aggfunc="mean")
ax = rates.plot(kind="bar", figsize=(9,6))
ax.set_title("Survival Was Strongly Shaped by Sex and Passenger Class",
             fontsize=15, weight="bold")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Survival rate")
ax.set_ylim(0,1)
ax.set_xticklabels(["1st","2nd","3rd"], rotation=0)
for c in ax.containers:
    ax.bar_label(c, fmt=lambda x:f"{x:.0%}", padding=3)
ax.legend(title="Sex")
plt.tight_layout()
plt.savefig(FIG/"01_survival_sex_class.png", dpi=220)
plt.close()

# 2. Age distribution by outcome
age_df = df.dropna(subset=["age"])
fig, ax = plt.subplots(figsize=(9,6))
sns.violinplot(data=age_df, x="survived", y="age",
               inner=None, cut=0, ax=ax)
sns.stripplot(data=age_df.sample(min(300,len(age_df)), random_state=42),
              x="survived", y="age", alpha=.25, size=3, ax=ax)
ax.set_title("Age Distribution of Survivors vs Non-Survivors",
             fontsize=15, weight="bold")
ax.set_xticklabels(["Did not survive","Survived"])
ax.set_xlabel("Outcome")
ax.set_ylabel("Age")
plt.tight_layout()
plt.savefig(FIG/"02_age_violin.png", dpi=220)
plt.close()

# 3. Age × class heatmap
d = df.copy()
d["age_band"] = pd.cut(
    d["age"], [0,10,20,30,40,50,60,80],
    labels=["0–10","11–20","21–30","31–40","41–50","51–60","61–80"]
)
heat = d.pivot_table(index="age_band", columns="pclass",
                     values="survived", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(heat, annot=True, fmt=".0%", cmap="Blues",
            vmin=0, vmax=1, linewidths=.5,
            cbar_kws={"label":"Survival rate"}, ax=ax)
ax.set_title("Survival Rate Changes Across Age and Class",
             fontsize=15, weight="bold")
ax.set_xlabel("Passenger class")
ax.set_ylabel("Age band")
plt.tight_layout()
plt.savefig(FIG/"03_age_class_heatmap.png", dpi=220)
plt.close()

# 4. Fare × age
scatter = df.dropna(subset=["age","fare"])
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=scatter, x="age", y="fare",
                hue="survived", style="pclass",
                alpha=.5, s=48, ax=ax)
ax.set_yscale("log")
ax.set_title("Fare, Age and Class Reveal Socioeconomic Separation",
             fontsize=15, weight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Fare (log scale)")
ax.legend(title="Survived / Class",
          bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG/"04_fare_age_scatter.png", dpi=220,
            bbox_inches="tight")
plt.close()

# 5. Fare ECDF
fig, ax = plt.subplots(figsize=(9,6))
for outcome, label in [(0,"Did not survive"),(1,"Survived")]:
    vals = np.sort(df.loc[df.survived==outcome, "fare"].dropna())
    y = np.arange(1,len(vals)+1)/len(vals)
    ax.plot(vals, y, linewidth=2.5, label=label)
ax.set_xscale("log")
ax.set_title("Cumulative Fare Distribution by Survival Outcome",
             fontsize=15, weight="bold")
ax.set_xlabel("Fare (log scale)")
ax.set_ylabel("Cumulative proportion")
ax.legend()
plt.tight_layout()
plt.savefig(FIG/"05_fare_ecdf.png", dpi=220)
plt.close()

print("Created five advanced visualizations.")

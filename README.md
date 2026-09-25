# Credit Risk Customer Segmentation

Unsupervised segmentation of loan applicants using K-Means clustering, exploring what natural customer groupings can reveal about credit risk, without a labeled default outcome.

## Overview

In my work as a Credit Analyst, I helped build a rule-based credit scoring model (the 5C framework) for a consumer financing product. That experience showed how much manual judgment is needed to assess risk when there's no historical default data to lean on yet — a common situation for a new financing product.

This project explores a different angle on the same problem: instead of manually defining scoring rules, can we let the data itself reveal meaningful applicant groupings?

## Problem Statement

Using the German Credit Data (1,000 loan applicants, no default/risk label available), can unsupervised clustering identify customer segments that could support early-stage screening decisions, before a full statistical credit scoring model exists?

## Dataset

**Statlog German Credit Data** (UCI / Kaggle) — 1,000 records, 9 features covering demographics, housing, financial accounts, and loan details.

Source: [Kaggle — German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit)

## Approach

1. **Data cleaning** — Missing values in `Saving accounts` and `Checking account` were identified as meaning "no account held," not random missingness, and encoded as an explicit "none" category rather than dropped or imputed.
2. **Feature engineering** — Ordinal encoding for ranked categories (savings/checking level, housing), binary encoding for sex.
3. **Standardization** — All features scaled so no single feature (e.g. credit amount, in the thousands) dominates the distance calculation used by clustering.
4. **Clustering** — K-Means tested for k = 2 to 7, with k = 4 selected using silhouette score, balanced against interpretability.
5. **Segment profiling** — Each cluster analyzed by average age, job level, credit amount, duration, housing, and sex to build a business-readable profile.

## Key Findings

| Segment | Size | Avg. Credit Amount | Avg. Duration | Notes |
|---|---|---|---|---|
| Established Male Homeowners | 467 | 2,241 DM | 17 mo | Majority homeowners |
| Established Female Homeowners | 187 | 2,057 DM | 16 mo | Majority homeowners |
| Young Renters | 161 | 2,480 DM | 17 mo | Youngest segment (avg. age 30), renters |
| **High-Value Borrowers** | 185 | **7,788 DM** | **38 mo** | Largest loans, longest terms, lowest savings cushion |

The **High-Value Borrowers** segment is the most actionable finding — larger loan exposure, longer repayment horizon, and the highest rate of holding no savings account. This combination would warrant closer scrutiny under a 5C-style framework, even before a full scoring model is built.

**Honest caveat:** sex turned out to be one of the strongest dividing lines in this clustering, separating "Established Male/Female Homeowners" into two segments. This is flagged deliberately — sex is not something that should be used as a basis for real credit decisions, and this is a useful reminder that unsupervised clustering can surface incidental correlations that aren't appropriate to act on directly.

Full write-up with narrative and charts: [link to blog post]

## Tech Stack

- Python (pandas, scikit-learn, matplotlib)
- K-Means clustering, PCA for visualization
- Google Colab

## Repository Structure

```
├── credit_segmentation.ipynb   # Full analysis notebook (code + outputs)
├── german_credit_data.csv      # Dataset
├── README.md
```

## Next Steps

- Validate segment stability using a different clustering algorithm (e.g. hierarchical clustering)
- If default labels become available, test whether the High-Value Borrowers segment actually correlates with higher default rates
- Explore feature importance more formally using an interpretable model (e.g. decision tree) as a companion analysis

---

**Author:** Luthfiyah Alifah Ridwan · [LinkedIn](https://www.linkedin.com/in/luthfiyahridwan/)

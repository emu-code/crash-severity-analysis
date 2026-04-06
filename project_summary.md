# Traffic Accident Severity Prediction: Summary Report

**Author:** Emumena Oweh  
**Date:** April 2026  
**Dataset:** US Accidents (Kaggle), 500,000-record sample  

## 1. Project Overview

This project analyzed traffic accident data to understand what factors are most associated with crash severity and to test whether severity can be predicted using machine learning.

The study combined data cleaning, feature engineering, exploratory analysis, and predictive modeling to answer two main questions:

1. What conditions are linked to more severe accidents?
2. How well can accident severity be predicted from available crash-related features?

---

## 2. Data and Preparation

The analysis began with a 500,000-record sample from the US Accidents dataset covering 2016 to 2023. The target variable was accident severity on a four-level scale:

- Severity 1: Minor  
- Severity 2: Moderate  
- Severity 3: Serious  
- Severity 4: Severe  

After cleaning missing values, removing weak or redundant columns, and treating skewed variables appropriately, the final dataset contained **445,073 records**.

New features were created to better capture real-world traffic patterns, including:

- Rush hour  
- Weekend indicator  
- Night driving  
- Clear weather  
- Season   
- High-risk junction  

This helped the models capture relationships that were not obvious from the raw variables alone. :contentReference[oaicite:1]{index=1}

---

## 3. Key Findings

### Clear Weather Paradox
One of the most striking findings was that severe accidents were more common in **clear or fair weather** than in fog. This suggests that drivers may travel faster when visibility is good, leading to higher-impact crashes.

### Infrastructure Matters
Accidents were generally more severe in locations without traffic signals or other control features. This was especially noticeable at conflict points such as junctions and intersections.

### Geography Matters
Location was a strong driver of severity. Latitude and longitude ranked among the top predictive features, showing that some regions or corridors consistently experience more severe crashes than others.

### Distance Was the Strongest Driver
The most important severity predictor was accident distance or road impact extent. Larger crash spread likely reflects higher-speed or more complex accident situations.

### Rush Hour Affected Frequency More Than Severity
Rush-hour periods showed more accidents overall, but severity levels did not vary as strongly by hour as expected. This suggests that time of day affects exposure and volume more than crash seriousness itself. :contentReference[oaicite:2]{index=2}

---

## 4. Models Trained

Four classification models were trained and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

Among them, **XGBoost** produced the best overall performance.

### Model Performance Summary

| Model | Accuracy | F1-Macro | ROC-AUC |
|------|----------|----------|---------|
| Logistic Regression | 64.9% | 0.352 | 0.717 |
| Decision Tree | 66.6% | 0.408 | 0.803 |
| Random Forest | 65.3% | 0.428 | 0.849 |
| XGBoost | 86.3% | 0.479 | 0.871 |

XGBoost achieved the highest overall accuracy, but Random Forest showed more balanced classification across rarer severity classes. This created an important trade-off between **overall accuracy** and **balanced severe-case detection**. :contentReference[oaicite:3]{index=3}

---

## 5. Main Insight from Model Comparison

The strongest overall model was **XGBoost**, but it leaned more heavily toward the majority severity class.

**Random Forest**, while less accurate overall, handled minority classes more evenly and performed better at detecting rare but important severe accidents.

This means the best model depends on the goal:

- **XGBoost** is better for general predictive performance  
- **Random Forest** is more useful when balanced severe-case detection matters more  

This was one of the most important lessons from the project: **high accuracy alone is not enough when the rarest classes are also the most critical**. :contentReference[oaicite:4]{index=4}

---

## 6. Top Severity Drivers

The five strongest severity drivers identified in the analysis were:

1. Distance  
2. Month  
3. Winter season  
4. Longitude  
5. Latitude  

Together, these show that severity is influenced by a mix of crash extent, seasonality, and geographic context rather than by a single factor alone. :contentReference[oaicite:5]{index=5}

---

## 7. Practical Recommendations

Based on the findings, the most important recommendations are:

- prioritize high-risk unsignalized intersections for safety upgrades  
- strengthen safety interventions during winter and low-visibility conditions  
- use geographic risk mapping to identify persistent severity hotspots  
- look beyond overall accuracy when choosing models for safety applications  
- develop public safety messaging that also addresses risky behavior during clear weather  

These recommendations reflect both the exploratory findings and the model interpretation stage. :contentReference[oaicite:6]{index=6}


## 8. Target Imbalance
The 500,000-record sample drawn from the original 7.7 million-record dataset was highly imbalanced, with approximately 77% of observations belonging to Severity 2. This skewed class distribution created a modeling challenge because high overall accuracy could be achieved simply by favoring the majority class. As a result, model evaluation did not rely on accuracy alone, but also considered macro F1-score, per-class recall, confusion matrices, and ROC-AUC to better assess performance across all severity levels.
---
## 9. Conclusion

This project showed that traffic accident severity can be predicted with strong overall performance and, more importantly, meaningfully interpreted.

The analysis found that severe accidents are linked to a combination of distance, season, geography, weather, and infrastructure context. It also revealed a valuable counterintuitive insight: severe crashes may be more common in clear weather than in fog, likely because of driver behavior.

Overall, the project demonstrates how machine learning can support transport safety not only by predicting accident severity, but also by helping identify where risks concentrate and where interventions may have the greatest impact.
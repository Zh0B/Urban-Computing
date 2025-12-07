# 🏙️ DMV Urban Computing Project  
## Price–Livability Mismatch Analysis (Ben’s Section)

This analysis investigates whether **housing prices in the DMV metropolitan area** accurately reflect the **livability** of each ZIP Code.

The goal is to answer:

> **Are 2025 housing prices aligned with actual livability?**  
> If not, which areas are *undervalued* or *overpriced*?

This component is part of a larger group project including:
- **Livability Index Construction (MCDA)** – by Eyan  
- **Lifestyle Clustering (Unsupervised Learning)** – by Wilson  
- **Price–Livability Mismatch Analysis (Residual Modeling)** – by Ben *(this section)*


---

# 📊 1. Data Overview

We analyze **313 ZIP Codes** across the DMV region after removing non-metropolitan areas.

### Datasets used:
- **DMV_House_Price_Data.csv**  
  - Median 2025 housing prices  
  - City / State / ZIP metadata  
- **Livability_Scores.csv** (from MCDA model)  
  - Transport Score  
  - Food Score  
  - Lifestyle Score  
  - Composite Livability Score (0–100)

These two datasets are merged via `ZipCode`.

---

# 🧮 2. Methodology

## **2.1 Regression Model**

To test whether livability explains housing price:


$\text{MedianPrice} = \beta_0 + \beta_1 \cdot \text{LivabilityScore}$


This is chosen because:
- Interpretable  
- Measures *expected* price based only on livability  
- Allows clean residual analysis

---

## **2.2 Residual Definition**


Residual is defined as:  
$\text{Residual} = \text{Actual Price} - \text{Predicted Price}$




Interpretation:

| Residual Value | Meaning | Market Insight |
|----------------|---------|----------------|
| **Residual < 0** | Undervalued | Good livability but cheaper than expected |
| **Residual > 0** | Overpriced | Lower livability but higher-than-expected price |

---

# 📈 3. Results

## **3.1 Price vs Livability Regression**

*(Insert your image here)*  
```md
![Price vs Livability](UC/price_vs_livability.png)

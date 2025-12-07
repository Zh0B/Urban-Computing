# Price–Livability Mismatch Analysis 

## **1 Regression Model**

To test whether livability explains housing price:

$\text{MedianPrice} = \beta_0 + \beta_1 \cdot \text{LivabilityScore}$

This is chosen because:
- Interpretable  
- Measures *expected* price based only on livability  
- Allows clean residual analysis

---

## **2 Residual Definition**

Residual is defined as:  
$\text{Residual} = \text{Actual Price} - \text{Predicted Price}$

Interpretation:

| Residual Value | Meaning | Market Insight |
|----------------|---------|----------------|
| **Residual < 0** | Undervalued | Good livability but cheaper than expected |
| **Residual > 0** | Overpriced | Lower livability but higher-than-expected price |

---

# 📈 3. Results

## Price vs Livability Regression
![Price vs Livability](UC/price_vs_livability.png)

## Price vs Housing Matrix
![Price vs Housing](UC/housing_matrix_residual_based.png)

## Residual Distribution
![Residual Distribution](UC/residual_distribution.png)
```md

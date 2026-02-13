# 𝗥𝗶𝗱𝗴𝗲 𝗥𝗲𝗴𝗿𝗲𝘀𝘀𝗶𝗼𝗻 𝘃𝘀 𝗟𝗶𝗻𝗲𝗮𝗿 𝗥𝗲𝗴𝗿𝗲𝘀𝘀𝗶𝗼𝗻

### 𝗣𝗿𝗮𝗰𝘁𝗶𝗰𝗮𝗹 𝗥𝗲𝗴𝘂𝗹𝗮𝗿𝗶𝘇𝗮𝘁𝗶𝗼𝗻 𝗘𝘅𝗽𝗲𝗿𝗶𝗺𝗲𝗻𝘁

---

## 📌 𝗢𝘃𝗲𝗿𝘃𝗶𝗲𝘄

This project explores how **Ridge Regression (L2 Regularization)** controls overfitting compared to standard Linear Regression.

Instead of only studying theory, both models were implemented, evaluated, and visualized to observe their behavior on real and synthetic data.

---

## 🎯 𝗢𝗯𝗷𝗲𝗰𝘁𝗶𝘃𝗲𝘀

✔ Train Linear Regression
✔ Apply Ridge Regression with multiple alpha values
✔ Compare performance using R² and RMSE
✔ Generate high-degree polynomial features to simulate overfitting
✔ Visualize how regularization smooths predictions

---

## 📊 𝗗𝗮𝘁𝗮𝘀𝗲𝘁

Diabetes dataset from **scikit-learn**

* 442 samples
* 10 medical features
* Target: disease progression score

---

## ⚙️ 𝗠𝗲𝘁𝗵𝗼𝗱𝗼𝗹𝗼𝗴𝘆

* Linear Regression
* Ridge Regression (L2 penalty)
* Polynomial Features (degree = 16)
* Train/Test split
* Evaluation metrics:

  * R² Score
  * RMSE

---

## 💻 𝗞𝗲𝘆 𝗜𝗺𝗽𝗹𝗲𝗺𝗲𝗻𝘁𝗮𝘁𝗶𝗼𝗻

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

model = Pipeline([
    ('poly', PolynomialFeatures(degree=16)),
    ('ridge', Ridge(alpha=20))
])
```

---

## 📈 𝗥𝗲𝘀𝘂𝗹𝘁𝘀 & 𝗢𝗯𝘀𝗲𝗿𝘃𝗮𝘁𝗶𝗼𝗻𝘀

### Linear vs Ridge

* Linear Regression → unstable with complex features
* Ridge → smoother and more stable predictions

### Effect of Alpha

* α = 0 → overfitting
* moderate α → best generalization
* large α → underfitting
<p align="center">
  <img src="/Plots.pngs/Plot comparing alphas.png" width=100% alt="Banner"/>
</p> 
---

## 🧠 𝗞𝗲𝘆 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴𝘀

* Complex models easily overfit
* Ridge shrinks weights to reduce variance
* Regularization improves generalization
* Same concept is used as **weight decay in deep learning**

---

## 🛠️ 𝗧𝗲𝗰𝗵 𝗦𝘁𝗮𝗰𝗸

* Python
* NumPy
* Matplotlib
* scikit-learn

---

## ▶️ 𝗛𝗼𝘄 𝘁𝗼 𝗥𝘂𝗻

```bash
pip install numpy matplotlib scikit-learn
python ridge_experiment.py
```

or open the notebook.

---

## 👩‍💻 𝗔𝘂𝘁𝗵𝗼𝗿

**Isha Maryam**
Undergraduate Computer Science • Machine Learning • Deep Learning & AI Enthusiast

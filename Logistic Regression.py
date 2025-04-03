#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy
from sklearn import linear_model
import matplotlib.pyplot as plt

#Reshaped for Logistic function.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


# In[17]:


logr = linear_model.LogisticRegression()
logr.fit(X,y)

#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(numpy.array([89]).reshape(-1,1))
print(predicted)


# In[ ]:


import numpy
from sklearn import linear_model


# In[ ]:


The `.reshape(-1,1)` function is crucial when preparing data for a logistic function (or any machine learning model in general). Let's break it down step by step.  

---

## **Why Use `.reshape(-1,1)`?**
### **1. Converts a 1D Array into a 2D Column Vector**
- In the given code:
  ```python
  X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
  ```
  - Initially, `X` is a **1D NumPy array** with shape `(12,)`.
  - `.reshape(-1,1)` transforms it into a **2D column vector** with shape `(12,1)`.
  
  **Before Reshape (1D Array):**
  ```
  [3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]
  Shape: (12,)
  ```

  **After Reshape (2D Column Vector):**
  ```
  [[3.78]
   [2.44]
   [2.09]
   [0.14]
   [1.72]
   [1.65]
   [4.92]
   [4.37]
   [4.96]
   [4.52]
   [3.69]
   [5.88]]
  Shape: (12,1)
  ```

---

### **2. Required Format for Machine Learning Models**
Most machine learning models in **scikit-learn**, including **Logistic Regression**, expect input features (`X`) to be a **2D array** of shape `(n_samples, n_features)`, even if there's only **one feature**.

- If we do **not** reshape:
  ```python
  X = numpy.array([3.78, 2.44, ...])  # Shape (12,)
  ```
  - This will cause an error when fitting a model like `LogisticRegression` because it expects `(12,1)`, **not** `(12,)`.

- If we **reshape to (12,1)**:
  ```python
  X = numpy.array([...]).reshape(-1,1)  # Shape (12,1)
  ```
  - Now, each sample has its own row, and logistic regression can process it correctly.

---

### **3. Ensures Compatibility with Vectorized Operations**
- Many machine learning models, including **Logistic Regression**, involve **matrix operations** like:
  \[
  y = \sigma (wX + b)
  \]
  - Here, `w` is the weight matrix of shape `(1,1)`, and `X` needs to be **2D** (`(12,1)`) for matrix multiplication to work properly.
  - If `X` remains **1D (`(12,)`)**, matrix operations will **fail** or give incorrect results.

---

### **4. Works Seamlessly with Pipelines & Transformations**
- Many preprocessing steps in `scikit-learn` (e.g., `StandardScaler`, `PolynomialFeatures`) also expect `X` to be **2D**.
- Reshaping ensures smooth integration with such transformations.

---

## **Summary**
✅ `.reshape(-1,1)` is used to convert a **1D array into a 2D column vector**, ensuring compatibility with:
1. **Machine learning models** (e.g., `LogisticRegression` in `scikit-learn`).
2. **Matrix operations** required for logistic function computation.
3. **Preprocessing techniques** like scaling and feature engineering.

🔹 Without `.reshape(-1,1)`, the model might throw an error or not function correctly. 🚀


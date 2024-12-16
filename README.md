# OptimizationMethods-MLP
# 🧠 A Multi-Layer Perceptron from scratch using Numpy for Age Regression
Final Project for Optimization Methods for Data Science

Author: Milad Torabi | Date: December 16, 2024

This repository contains the final project for the Optimization Methods for Data Science course, focusing on building and training a Multi-Layer Perceptron (MLP) for predicting age using a given set of features. The project demonstrates the use of L-BFGS-B optimization, efficient weight initialization, and hyperparameter tuning to achieve optimal results.

# 📄Report
The complete project report is available for detailed reference:
[📘 View Full Report (PDF)](./Report_MLP.pdf)

# 📚Table of Contents
1-Problem Description
2-Data Preprocessing
3-Optimization Goal and Strategy
4-Neural Network Architecture and Training Details
5-Hyperparameter Tuning
6-Optimization Details
7-Final Model Training and Results
8-Conclusion

# 🧠 Multi-Layer Perceptron for Age Regression

**Project for Optimization Methods for Data Science**  
**Author:** Milad Torabi | **Date:** December 16, 2024  

This project develops a **Multi-Layer Perceptron (MLP)** to predict age from input features using an **L2-regularized MSE loss**. The model was optimized using **L-BFGS-B**, and hyperparameters were fine-tuned to achieve optimal performance.

---

## 📄 **Project Report**
Access the full project report for detailed explanations:  
[📘 View Report (PDF)](Report_11_MLP.pdf)  

*(Make sure the `Report_11_MLP.pdf` file is located in the root of your repository.)*  

---

## 📋 **Key Highlights**
- **Objective**: Age prediction using a Multi-Layer Perceptron (MLP).  
- **Dataset**: Custom **AGE_REGRESSION.csv** with 32 input features and 1 target (age).  
- **Loss Function**: Mean Squared Error (MSE) with L2 Regularization.  
- **Architecture**: 32 → 128 → 64 → 1 (fully connected layers).  
- **Optimization**: **L-BFGS-B** with gradient clipping and pre-activation clipping.  
- **Best Hyperparameters**:  
  - Hidden Layers: [128, 64]  
  - Activation Function: Tanh  
  - Regularization (λ): 0.01  
  - Initialization: He  

---

## 📚 **Project Structure**
📂 Project Root  
├── 📘 Report_11_MLP.pdf (Full Project Report)  
├── 📄 README.md (This File)  
├── 📊 AGE_REGRESSION.csv (Dataset)  
├── 📂 Functions_MLP.py (Code Files)  
└── 📂 run_MLP.ipynb (Jupyter Notebooks for Training)  


---

## 📈 **Performance**
| **Metric**         | **Train**         | **Validation**    | **Test**         |
|-------------------|-------------------|-------------------|------------------|
| **Initial Loss**   | 1681.80           | -                 | -                |
| **Final Loss**     | 99.07             | -                 | 91.31            |
| **Initial MAPE**   | 99.70             | -                 | -                |
| **Final MAPE**     | 22.93             | 23.06             | 23.56            |

---

## 📚 **Methodology**
1. **Data Preprocessing**:  
   - Features were Z-score normalized for better convergence.  
   - The dataset was split into features (X) and target (age).  

2. **MLP Architecture**:  
   - 2 Hidden Layers: 128, 64 Neurons (Tanh Activation)  
   - Custom Backpropagation Implementation  
   - Weight Initialization: **He** for Tanh activation  

3. **Optimization**:  
   - **L-BFGS-B**: Converged from initial loss 1681.80 to final loss 91.31.  
   - **Stopping Criteria**: 200 iterations, tolerance = \(10^{-4}\).  

📧 Contact
For any questions or suggestions, feel free to reach out:
Milad Torabi
📧 [miladtorabi65@gmail.com]




# Elevate Labs - AI & ML Internship

## Task 5: Decision Trees and Random Forests

### Objective

The objective of this task is to understand and implement tree-based machine learning models for classification using Decision Tree and Random Forest algorithms. The models are trained, evaluated, and compared using the Heart Disease dataset.

---

## Tools Used

- Python
- Visual Studio Code
- Pandas
- Matplotlib
- Scikit-learn
- Graphviz (for tree visualization)

---

## Dataset

- Heart Disease Dataset (`heart.csv`)

---

## Steps Performed

### 1. Imported Required Libraries
Imported the necessary Python libraries for data handling, visualization, model training, and evaluation.

### 2. Loaded the Dataset
Loaded the Heart Disease dataset into a Pandas DataFrame.

### 3. Explored the Dataset
- Displayed the first five rows.
- Checked dataset information.
- Verified data types and missing values.

### 4. Selected Features and Target
- Used all columns except **target** as input features.
- Used the **target** column as the output variable.

### 5. Split the Dataset
Divided the dataset into:
- 80% Training Data
- 20% Testing Data

using Scikit-learn's `train_test_split()` function.

### 6. Trained a Decision Tree Classifier
Built a Decision Tree model using the training dataset and generated predictions for the test dataset.

### 7. Visualized the Decision Tree
Created a graphical representation of the Decision Tree to understand how decisions are made.

### 8. Controlled Overfitting
Limited the maximum tree depth using the `max_depth` parameter and compared the model's performance before and after controlling tree depth.

### 9. Trained a Random Forest Classifier
Built a Random Forest model using multiple Decision Trees and generated predictions on the test dataset.

### 10. Compared Model Accuracy
Compared the accuracy of:
- Decision Tree Classifier
- Random Forest Classifier

to determine which model performs better.

### 11. Interpreted Feature Importance
Calculated feature importance scores to identify which features contribute the most to heart disease prediction.

### 12. Performed Cross Validation
Evaluated the Random Forest model using 5-fold Cross Validation to obtain a more reliable estimate of model performance.

---

## Model Evaluation

The following evaluation methods were used:

- Decision Tree Accuracy
- Decision Tree Accuracy (Controlled Depth)
- Random Forest Accuracy
- Feature Importance
- 5-Fold Cross Validation
- Average Cross Validation Accuracy

---

## Files Included

- task5.py
- heart.csv
- decision_tree.png
- README.md

---

## Results

Successfully implemented Decision Tree and Random Forest classifiers on the Heart Disease dataset.

The Decision Tree model was visualized and analyzed for overfitting by controlling its depth. The Random Forest model achieved improved performance by combining multiple decision trees. Feature importance analysis identified the most influential features, and cross-validation provided a reliable estimate of model accuracy.

---

## Learning Outcomes

- Learned how Decision Trees perform classification.
- Understood tree visualization and decision-making.
- Learned how overfitting occurs and how to control it using tree depth.
- Implemented Random Forest and compared its performance with Decision Tree.
- Interpreted feature importance scores.
- Evaluated machine learning models using Cross Validation.

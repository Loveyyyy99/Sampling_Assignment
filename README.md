# Sampling Assignment

## Objective
To study the impact of different sampling techniques on machine learning models when working with a highly imbalanced credit card dataset.

## Dataset
- Creditcard_data.csv
- Binary classification dataset
- Highly imbalanced classes

## Steps Followed
1. Loaded and analyzed the imbalanced dataset  
2. Converted the dataset into a balanced dataset using SMOTE  
3. Created five samples from the balanced dataset  
4. Applied five different sampling techniques  
5. Trained five different machine learning models  
6. Compared model accuracy and visualized results  

## Dataset Balancing
- Technique used: SMOTE
- Result: Equal number of samples in both classes
- Output file: data/balanced_data.csv

## Samples Created
Five samples were created from the balanced dataset:
- sample_1.csv
- sample_2.csv
- sample_3.csv
- sample_4.csv
- sample_5.csv

## Sampling Techniques
- Sampling1: Random Under Sampling  
- Sampling2: Random Over Sampling  
- Sampling3: SMOTE  
- Sampling4: SMOTE + ENN  
- Sampling5: SMOTE + Tomek Links  

## Machine Learning Models
- M1: Logistic Regression  
- M2: Decision Tree  
- M3: Random Forest  
- M4: Support Vector Machine  
- M5: K-Nearest Neighbors  

## Results
- Accuracy was calculated for each Sampling–Model combination
- Results saved in: results/accuracy_table.csv

## Visualizations
- Accuracy heatmap
- Model-wise comparison bar plot
- Sampling-wise comparison bar plot

All visualizations are saved in the results folder.

## Conclusion
Sampling techniques have a significant impact on model performance. Oversampling and hybrid sampling techniques generally performed better, and ensemble models showed strong accuracy on the balanced dataset.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the notebook:
   notebooks/sampling_analysis.ipynb

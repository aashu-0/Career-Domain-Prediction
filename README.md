# Preferred Career Domain Predictor

A machine learning-based tool to help students discover suitable career domains based on their academic background, interests, and skillsets.

## Team
- Aashutosh Mishra
- Gaurav Upadhyay

## Problem Statement
can we predict a student's preferred career domain using personal and academic attributes?  
the goal is to recognize patterns and recommend likely-fit career domains—not to prescribe exact outcomes.

## Dataset
- collected via google form (277 valid responses)  
- features: cgpa, department, skill ratings, career preferences, influences, higher studies, etc.  
- cleaned and transformed into 16 relevant features

## Models and Accuracy
| model               | accuracy | precision | recall | f1-score |
|--------------------|----------|-----------|--------|----------|
| random forest       | 0.73     | 0.77      | 0.73   | 0.74     |
| mlp (neural net)    | 0.71     | 0.74      | 0.71   | 0.72     |
| knn classifier      | 0.70     | 0.80      | 0.70   | 0.72     |
| catboost            | 0.70     | 0.75      | 0.70   | 0.69     |
| xgboost             | 0.62     | 0.64      | 0.62   | 0.62     |

Random forest performed best in terms of accuracy and interoperability.

## Try it out
- Deployed app: [demo](https://preferredcareer.streamlit.app/)

## Key Learnings
- complete ml pipeline implementation  
- real-world data handling and transformation  
- model selection and evaluation  
- web app deployment using streamlit

## Limitations
- small sample size  
- subjective survey responses  
- no actual career outcome data  
- limited diversity in dataset

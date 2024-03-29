# Predicting Cardiovascular Diseases

<img src="https://www.augustahealth.com/sites/default/files/blog/hands-cupped-around-a-heart.jpg" alt="drawing" width="100%"/>

_This project uses synthetic data from Kaggle that contains patient records across 12 features. The dataset can be found on [Kaggle - Cardiovascular Disease dataset](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset). Header image retrieved [here.](https://www.augustahealth.com/sites/default/files/blog/hands-cupped-around-a-heart.jpg)_

In this project, we will explore a classic classification problem: **to predict whether a patient has cardiovascular disease or not based on 70,000 patient records.** A business situation will be set to guide our analysis throughout the project.

This readme will contain the following points:
1. How to tackle a binary classification problem;
2. Explore classification algorithms and understand their differences;
3. A summary of the main insights obtained from the data from a business perspective;

**The project management methodology** used for this classification project was the **Cross-Industry Standard Process for Data Mining (CRISP-DM)**, and the outcomes presented in this readme reflects the 2nd CRISP-DM cycle with satisfactory results.

### How to read this README.MD?

1. If you wish to **check the codes for this project**, access the Jupyter notebook [here](https://github.com/alanmaehara/cardiovascular-analysis/blob/master/cardiovascular-diseases-002.ipynb);
2. If you wish to **read the project's main findings instead of going through the entire project**, look no further and [get there](#main-findings);
3. A [(go to next section)]() hyperlink will be available for each section to make your reading smoother, and a [(skip theory)]() hyperlink will be there to skip technical explanations. 

I would appreciate any comments or suggestions on how to improve this project. Please feel free to add me on GitHub or [Linkedin](https://www.linkedin.com/in/ammaehara/) - I will get back to you as soon as possible. 

With no further due, let's get started!

---
## Table of Contents
- [A brief introduction to CVDs](#a-brief-introduction-to-cvds)
- [Main Findings](#main-findings)
- [01. The Business Problem and Deliverables](#01-the-business-problem-and-deliverables)
- [02. Data Preparation and Feature Engineering](#02-data-preparation-and-feature-engineering)
- [03. Exploratory Data Analysis (EDA)](#03-exploratory-data-analysis-eda)
- [04. Data Preprocessing and Feature Selection](#04-data-preprocessing-and-feature-selection)
- [05. Machine Learning Modeling](#05-machine-learning-modeling)
- [06. Hyperparameter Tuning](#06-hyperparameter-tuning)
- [07. Business Performance](#07-business-performance)

---
## A brief introduction to CVDs

[(go to next section)](#main-findings)

Before we jump to the business problem, a brief introduction to cardiovascular diseases (CVDs) is of help. According to [heart.org](https://www.heart.org/en/health-topics/consumer-healthcare/what-is-cardiovascular-disease), CVDs can refer to a number of conditions:

- **Heart Disease**: this is commonly connected with a process called _atherosclerosis_ - which is a condition that a substance called "plaque" builds up in the arteries' walls. Once the walls are blocked, a heart attack or a stroke can happen;
- **Heart Attack**: this is the consequence of heart disease. When a heart attack happens, part of the heart muscle that is supplied by the blocked artery begins to die.
- **Stroke**: there are two types of strokes, ischemic and hemorrhagic. An ischemic stroke occurs when a blood vessel that feeds the brain gets blocked. Depending on the blood vessel affected, this can lead to the loss of the brain functions pertaining to that area. A hemorrhagic stroke occurs when a blood vessel within the brain bursts. One of the factors that contribute to hemorrhagic stroke is high blood pressure - hypertension.
- **Heart Failure or Congestive heart failure**: this is a condition where the heart isn't pumping blood well due to inappropriate levels of oxygen and blood.
- **Arrhythmia**: this is a condition of an abnormal heart rhythm. There are two types of arrhythmia: Bradycardia (too slow heart rate - less than 60 BPM) and Tachycardia (too fast heart rate - more than 100 BPM);
- **Heart Valve Problems**: a type of problem related to the openness of heart valves. If valves aren't opened enough to allow the blood to flow, or don't close enough, or still the valve prolapses (change its position), then a heart valve problem is identified.

#### Causes of CVDs

According to the [United Kingdom National Health Service (NHS)](https://www.nhs.uk/conditions/cardiovascular-disease/), the exact causes of CVD aren't clear but there are risk factors that increase the chance of getting a CVD:

- **High Blood Pressure**: one of the most important risk factors for CVD. If your blood pressure is high, it can lead to damages in your blood vessels;
- **Smoking**: tobacco substances can also lead to damages in the blood vessels;
- **High Cholesterol**: high LDL (Low-Density Lipoproteins) cholesterol levels can narrow blood vessels and increase the chance of developing blood clots;
- **Diabetes**: a disease that causes one's blood sugar level to increase. High blood sugar levels can also damage blood vessels, which in turn make blood vessels narrower;
- **Inactivity**: exercising regularly is one of the factors that contribute to a healthy heart. The lack of physical exercise increases the likelihood of developing high blood pressure, cholesterol, and obesity - all risk factors of CVDs;
- **Obesity**: being obese or overweight can increase the risk of developing diabetes or high blood pressure, which are risk factors of CVDs. If your Body Mass Index (BMI) is 25 or above, you are at risk of developing CVDs - or if your waist is above 94cm (37 inches) for men, 80cm (31.5 inches) for women;
- **Family History of CVD**: high risk of having CVD is associated with a family history of CVD. If your father or brother were diagnosed with CVD before the age of 55, or your mother or sister were diagnosed with CVD before the age of 65, you have a family history of CVD;
- **Ethnic Background**: CVD is more common in people that have an association with high blood pressure or type 2 diabetes with their ethnic background. For instance, in the UK is common that people from South Asia, Africa, or the Caribbean have an increased likelihood of getting CVD;
- **Age**: CVD is most common in people over 50;
- **Gender**: men are more likely to develop CVD at an earlier age than women;
- **Diet**: unhealthy diet habits can lead to high blood pressure and high cholesterol, which in turn leads to a high risk of getting CVDs;
- **Alcohol**: excessive consumption of alcohol can lead to high blood pressure and high cholesterol.

Since hypertension is one of the main risk factors for CVD, it is important to know the threshold values that characterize a person's blood pressure level. We interpret blood pressure levels by using two values: systolic and diastolic. According to [healthline.com](https://www.healthline.com/health/diastole-vs-systole#TOC_TITLE_HDR_1):

> Your **systolic blood pressure is the top number on your reading**. It measures the force of blood against your artery walls while your ventricles — the lower two chambers of your heart — squeeze, pushing blood out to the rest of your body.

> Your **diastolic blood pressure is the bottom number on your reading**. It measures the force of blood against your artery walls as your heart relaxes and the ventricles are allowed to refill with blood. Diastole — this period of time when your heart relaxes between beats — is also the time that your coronary artery is able to supply blood to your heart.

A useful chart from [ESC Guidelines 2013](https://slideplayer.com/slide/7436245/) can be of help to interpret blood pressure:

![](img/classification.PNG)
&nbsp;

[back to top](#table-of-contents)

---
## Main Findings

[(go to next section)](#01-the-business-problem-and-deliverables)

**Note: if you plan to read the entire project, skip this section.**

In this project, a classification model is built for a fictional healthcare business specialized in cardiovascular disease (CVD) detection called **Cardio Catch Diseases**. The goal is twofold: (1) to understand the main causes for the presence/absence of CVDs; (2) to increase and stabilize the current diagnosis' precision rate, which varies between 55% and 65% due to the diagnosis' complexity and operation inefficiencies.

Operation costs per diagnosis are around \$1,000.00, and the client pays \$500.00 for every 5% increase in diagnosis precision rate. If diagnostic accuracy is 50% or below, the customer does not pay for the service.

The data utilized in this project contains 70,000 data points from patient records across 12 features:

- Age
- Height
- Weight
- Gender
- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol
- Glucose
- Smoking
- Alcohol intake
- Physical activity
- **Presence or absence of CVD (target)**

We divide the main findings into two: (1) understanding the main causes of CVD for the firm's patient; (2) Machine learning solution and business performance.

### 1. Main Causes of CVD for the Firm's Patients

The following mindmap contains the main drivers for developing CVDs: 

![](img/hypothesis.jpg)

Data related to CVD's family history, ethnic background, saturated fat intake, salt intake, fiber/grain food/fruit/vegetable consumption are not present in the dataset. In a non-fictional situation, such data (and much more) would be asked to the business.

A summarized description of the project's dataset and some key findings are presented below:

* **General Information**

  - The **shortest person** is 1,25m tall and the **tallest person** is 2,07m tall.
  - The **fattest person** weighs 180kg and the **thinnest person** weighs 34kg.
  - The **youngest person** is 30 years old and **the oldest person** is 65 years old.
  - Patients in the dataset present an elevated high blood pressure on average.

* **Target Variable**
  - Number of patients with CVD: 33,171
  - Number of patients without CVD: 32,292

![](img/target.PNG)

* **Numerical variables (histograms)**: 

  - **Age**: there is a concentration of patients aged 50 to 60;
  - **Height**: distribution concentrates around the height of 160cm;
  - **Weight**: distribution is slightly shifted to the left. Data concentrated around 70kg;
  - **Systolic Pressure (ap_hi) & Diastolic Pressure (ap_lo)**: Systolic and Diastolic measurements concentrated around 120/80mmHg (normal blood pressure);
  - **BMI**: distribution is slightly shifted to the left, concentrated around 25; 

![](img/num_var.PNG)

&nbsp;
* **Categorical variables (barplots)**

  - **Gender**: There are almost twice patients of gender 1 than 2;
  - **Cholesterol**: Number of patients with "well-above" cholesterol levels are 5x bigger than others;
  - **Glucose**: Number of patients with "well-above" glucose levels are 5x bigger than others;
  - **Smoke**: Number of non-smoker patients is 6x bigger than smokers;
  - **Alcohol intake**: Number of patients who aren't alcohol consumers are 5x bigger than patients who are alcohol consumers;
  - **Active**: Number of physically active patients are almost 5x bigger than sedentary patients;
  - **Systolic**: Around 60% of patients have normal systolic levels. Grade 1 Hypertension patients make up for around 20% of patients, followed by "High Normal" (15%), Grade 2 Hypertension, and Grade 3 Hypertension patients;
  - **Diastolic**: Around 70% of patients have normal systolic levels. Grade 1 Hypertension patients make up for around 20% of patients, followed by Grade 2 Hypertension, "High Normal", and Grade 3 Hypertension patients;
  - **Systolic + Diastolic**: Around 50% of patients have normal blood pressure levels. Grade 1 Hypertension patients make up for around 20% of patients, followed by "Optimal", "High-Normal", Grade 2 Hypertension, "Isolated", and Grade 3 Hypertension patients;
  - **Hypertension**: Around 60% of patients don't have hypertension;
  - **BMI**: Around 70% of patients have normal or overweight BMI levels. Obesity Class 1's patients make up for around 20%, followed by Obesity class 2, class 3, and underweight.
  - **CVD Risk Scale**: The scale suggests that around 90% of patients have a very low or low risk of having CVD. 

![](img/cat1.PNG)
![](img/cat2.PNG)


* **Correlation of variables with the target variable**

  - Mild positive correlation with **systolic**, **diastolic**, and **hypertension (systolic + diastolic pressures)**. 
  - Weak positive correlation with **BMI**, **cholesterol**, and **CVD Risk**. CVD Risk is a variable that measures CVD risk with a 0-10 scale, being zero equal to a low risk of having CVD and 10 otherwise; 
  - Zero or almost zero correlation with **gender**, **glucose**, **smoking**, **alcohol intake**, and **physical activity**. 

According to the insights earned above, hypotheses were derived and tested. For more details on each hypothesis, check the [project's notebook](https://github.com/alanmaehara/cardiovascular-analysis/blob/master/cardiovascular-diseases-002.ipynb):

|Hypothesis|Verdict|
|--|--|
|1. More than 50% of patients with CVD are of gender 1.|True|
|2. The proportion of CVD cases surpass of non-CVD cases in patients weighting above 90kg.|False|
|3. The proportion of CVD cases surpass of non-CVD cases in patients aged 50 or above.|False|
|4. The average height among patients with CVD and without CVD is different with a confidence interval of 95%|True|
|5. More than 70% of patients with CVD are patients who smoke.|False|
|6. More than 70% of patients with CVD are physically inactive.|False|
|7. More than 70% of patients with CVD are under the "well above" glucose level category.|True|
|8. At least 30% of patients with CVD are patients who drink alcohol.|False|
|9. At least 30% of patients with CVD are under the "well above" cholesterol level category.|True|
|10. The average BMI among patients with CVD and without CVD is different with a confidence interval of 95%|True|
|11. More than 50% of patients have hypertension (Grade 1 Hypertension or above)|True|
|12. More than 70% of patients with CVD have a CVD Risk Scale of "high" or above.|False|

### 2. Machine Learning Solution + Business Performance

In this project, 11 machine learning models were tested in order to: (1) find the best performer model; (2) to learn the behavior of most common classification models to this type of problem. We have obtained the following results:

![](img/modeling.PNG)

By using the F1-Score as the main judge, and running a thorough analysis of confusion matrices, ROC-AUC curves, probability distributions, and cross-validation with holdout data, the predictive model chosen was the LGBM model. The final result is as follows:

![](img/main_findings.PNG)

The model we chose generates CVD predictions with precision between 76.03% and 75.03%. Since there are more than 70,000 patients in the dataset, we can calculate the firm's current profit based on the current solution (healthcare software) performance, and compare it with the machine learning solution we have built for this project. 

Important notes:
- For each 5\% increase in precision rate above 50\%, **there is a \$500 increase for the patients' bill**;
- For unitary percent increase, the patient's bill increase accordingly ($100);
- The diagnosis cost is **\$1000 per patient**;
- For the existing solution, the precision rate varies from **55% (worst scenario) and 65% (best scenario)**;
- For the LGBM model, the precision rate varies from **75.03% (worst scenario) and %76.03 (best scenario)**

Total profit was:

![](img/business.PNG)

For 70,000 patients, **the current operation would have a debt of ~\$35 million** in the worst scenario **and would have a profit of ~\$35 million** in the best scenario.

**Under the model built in this project, the firm would never see a negative value**: in the worst scenario, **profit is around \\$105,2 million**; in the best scenario, **the profit would be \$112,2 million**. 

Considering the best scenario possible, **the machine learning model would increase revenues to the order of 68\%** compared with the current software solution. 

[back to top](#table-of-contents)

---
## 01. The Business Problem and Deliverables
[(go to next section)](#02-data-preparation-and-feature-engineering)

**Cardio Catch Diseases** is our fictional healthcare business specialized in cardiovascular disease (CVD) diagnostic services. **The firm's revenue streams come only from CVD diagnostics**; therefore, some of the ways the firm can increase its profit are by:

1. Adding complementary services to the existing one;
2. Increasing the customer base size;
3. Increasing the frequency of diagnostics per patient;
4. Raising service price
5. Reducing fixed and variable costs;
6. Increasing the number of sponsors;
7. Improve service quality and use performance-based pricing strategy 

Currently, the diagnosis of cardiovascular diseases is made manually by a team of specialists. **The current precision rate of the diagnosis varies between 55% and 65%**, due to the diagnosis' complexity and also the fatigue of the team who take turns to minimize the operational risks during the procedures. 

**The firm's price strategy is set as performance-based**: the better performance on diagnostics, the higher the price charged by the firm.

Details:
- **The cost of each diagnosis (including the devices and the payroll of the analysts) is around $1,000.00.** The price tag for the service varies according to the diagnosis precision achieved by the team of specialists. 

- **The client pays 500.00 for every 5% increase in diagnosis precision rate above 50%**. For example, for a precision rate of 55%, the diagnosis costs 500.00 for the client; while for a rate of 60%, the value is 1,000.00.

- **If the diagnostic accuracy is 50% or below, the customer does not pay for the service**.

**The firm's challenge is to stabilize the diagnostic's precision rate** in order to get more control over the future cash-flows of the firm. Since each diagnostic precision rate is unstable, the company wishes to use a more sophisticated tool than the existing one, which is a healthcare software solution that calculates precision according to some unknown thresholds on each of the following variables:

|Variable|Feature Type|Variable Name|Data Type|
|--|--|--|--|
|Age | Objective Feature | age | numerical, discrete (days)|
|Height | Objective Feature | height | numerical, continuous (cm) |
|Weight | Objective Feature | weight | numerical, continuous (kg) |
|Gender | Objective Feature | gender | categorical, binary (labels are unknown) |
|Systolic blood pressure | Examination Feature | ap_hi | numerical, continuous (mmHg) |
|Diastolic blood pressure | Examination Feature | ap_lo | numerical, continuous (mmHg) |
|Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
|Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
|Smoking | Subjective Feature | smoke | categorical, binary  |
|Alcohol intake | Subjective Feature | alco | categorical, binary |
|Physical activity | Subjective Feature | active | categorical, binary |
|Presence or absence of cardiovascular disease | Target Variable | cardio | categorical, binary |

where:
- Objective: factual information;
- Examination: results of medical examination;
- Subjective: information given by the patient.

_Assumption: except for gender, we will treat binary variable values as `1 = existence` and `0 = nonexistence` of the respective variable trait._ 

### Deliverables

A dataset of 70,000 past patients with information of all the variables shown in the table will be used in this project. The solution will be the creation of a model that predicts the existence of cardiovascular diseases on patients **by a precision rate of at least higher than 65%.** 

[back to top](#table-of-contents)

---
## 02. Data Preparation and Feature Engineering
[(go to next section)](#03-exploratory-data-analysis-eda)

The dataset was split into training (67% of the entire data), test (20%), and validation data (13%). Data were split in the Data Preprocessing section for manipulation convenience. No missing values were found.

#### Data Dimensions (rows x columns)

  * Train dataset: 44,514 x 18 
  * Test dataset: 13,093 x 18
  * Valid dataset: 7,856 x 18

#### Outlier Analysis

Outliers for numerical variables were studied and removed accordingly (around 6.48% of the entire data). For more details on how the outliers were identified and removed, please access the [notebook.](https://github.com/alanmaehara/cardiovascular-analysis/blob/master/cardiovascular-diseases-002.ipynb)

In practice, one would consult the firm's business/medical team before removing outliers from the dataset. In this project, we made assumptions according to the medical literature as we identify outliers.

#### Descriptive Statistics

![](img/descriptive.PNG)

**Highlights**:
- **The shortest person** is 1,25m tall and **the tallest person** is 2,07m tall.
- **The fattest person** weighs 180kg and the **thinnest person** weighs 34kg.
- **The youngest person** is 30 years old and **the oldest person** is 65 years old.
- Patients in the dataset present **an elevated high blood pressure** on average.

#### Feature Engineering

In order to guide our feature engineering process (and later on, the [exploratory data analysis](#03-exploratory-data-analysis-eda)), a mindmap that connects the risk factors of CVD was created.

![](img/hypothesis.jpg)

&nbsp;

Based on the available data, the new features created were:

- `systolic`: a categorical variable that indicates the systolic level of the patient according to the [ESC Guidelines 2013](https://slideplayer.com/slide/7436245/) chart.
- `diastolic`: a categorical variable that indicates the diastolic level of the patient according to the [ESC Guidelines 2013](https://slideplayer.com/slide/7436245/) chart.
- `sys_diast`: a categorical variable that indicates both the systolic and diastolic levels of the patient according to the [ESC Guidelines 2013](https://slideplayer.com/slide/7436245/) chart.
- `hyper`: A binary variable indicating 1 = presence of hypertension; 0 = otherwise. If at least one of the above new variables indicates that the patient has high blood pressure, the patient will receive 1.
- `BMI`: numerical, continuous variable indicating the body mass index (BMI) of the patient.
- `cvd_risk`: a variable that measures CVD risk with a 0-10 scale. If zero, patients have a very low risk of having CVD; 10 represents the opposite. The variable is a sum of the categorical variables glucose, cholesterol, smoke, alcohol, active, hyper, and BMI.

[back to top](#table-of-contents)

## 03. Exploratory Data Analysis (EDA)

[(go to next section)](#04-data-preprocessing-and-feature-selection)

### I. Univariate Analysis 

* **Target Variable (cardio)**: Classes are balanced and won't need any balancing technique.
  - Number of patients with CVD: 33,171
  - Number of patients without CVD: 32,292

![](img/target.PNG)

* **Numerical variables (histograms)**: 

![](img/num_var.PNG)

Highlights:
- **Age**: there is a concentration of patients aged 50 to 60;
- **Height**: distribution concentrates around a height of 160cm in a normal shape;
- **Weight**: distribution is slightly shifted to the left. Data concentrated around 70kg;
- **Systolic Pressure (ap_hi) & Diastolic Pressure (ap_lo)**: Systolic and Diastolic measurements concentrated around 120/80mmHg - normal blood pressure;
- **BMI**: distribution is slightly shifted to the left, concentrated around 25; 
  
&nbsp;
* **Categorical variables (barplots)**

![](img/cat1.PNG)
![](img/cat2.PNG)

Highlights:
- **Gender**: There are almost twice patients of gender 1 than 2;
- **Cholesterol**: Number of patients with "well-above" cholesterol levels are 5x bigger than others;
- **Glucose**: Number of patients with "well-above" glucose levels are 5x bigger than others;
- **Smoke**: Number of non-smoker patients is 6x bigger than smokers;
- **Alcohol**: Number of patients who aren't alcohol consumers are 5x bigger than patients who are alcohol consumers;
- **Active**: Number of physically active patients are almost 5x bigger than sedentary patients;
- **Systolic**: Around 60% of patients have normal systolic levels. Grade 1 Hypertension patients make up for around 20% of patients, followed by "High Normal" (15%), Grade 2 Hypertension, and Grade 3 Hypertension patients;
- **Diastolic**: Around 70% of patients have normal systolic levels. Grade 1 Hypertension patients make up for around 20% of patients, followed by Grade 2 Hypertension, "High Normal", and Grade 3 Hypertension patients;
- **Systolic + Diastolic**: Around 50% of patients have normal blood pressure levels. Grade 1 Hypertension patients make up for around 20% of patients, followed by "Optimal", "High-Normal", Grade 2 Hypertension, "Isolated", and Grade 3 Hypertension patients;
- **Hypertension**: Around 60% of patients don't have hypertension;
- **BMI**: Around 70% of patients have normal or overweight BMI levels. Obesity Class 1's patients make up for around 20%, followed by Obesity class 2, class 3, and underweight.
- **CVD Risk Scale**: The scale suggests that around 90% of patients have a very low or low risk of having CVD. 

&nbsp;
### II. Bivariate Analysis

Twelve hypotheses were derived from the mindmap and were examined against the target variable. Here we highlight three main hypotheses - if you wish to check for the complete list of hypotheses, see the [notebook](https://github.com/alanmaehara/cardiovascular-analysis/blob/master/cardiovascular-diseases-002.ipynb).


#### H4. The average height among patients with CVD and without CVD is different with a confidence interval of 95% 

![](img/h4.PNG)
![](img/h4_1.PNG)
![](img/h4_2.PNG)

**TRUE**: Since the p-value is less than the alpha value (0.05), the null hypothesis is rejected. In other words, **the average height of patients with CVD and the average height of patients without CVD are different.**

#### H10. The average BMI among patients with CVD and without CVD is different with a confidence interval of 95% 

![](img/h10.PNG)
![](img/h10_1.PNG)

**TRUE**: The p-value is very close to zero and lower than the alpha value (0.05). Since the mean BMI for patients without CVD is not in between the confidence interval, we reject the null hypothesis. Therefore, **BMI is relevant to determine whether a patient has a CVD or not**.

#### H11. More than 50% of patients have hypertension (Grade 1 Hypertension or above)

![](img/h11.PNG)

**TRUE**: **More than 50% of patients with CVD have hypertension.** CVD Patients without hypertension make up for \~43% of the dataset. Among patients without CVD, the proportion of hypertensive patients are lower than non-hypertensive patients.

#### Summary of Hypotheses

|Hypothesis|Verdict|
|--|--|
|1. More than 50% of patients with CVD are of gender 1.|True|
|2. The proportion of CVD cases surpass of non-CVD cases in patients weighting above 90kg.|False|
|3. The proportion of CVD cases surpass of non-CVD cases in patients aged 50 or above.|False|
|4. The average height among patients with CVD and without CVD is different with a confidence interval of 95%|True|
|5. More than 70% of patients with CVD are patients who smoke.|False|
|6. More than 70% of patients with CVD are physically inactive.|False|
|7. More than 70% of patients with CVD are under the "well above" glucose level category.|True|
|8. At least 30% of patients with CVD are patients who drink alcohol.|False|
|9. At least 30% of patients with CVD are under the "well above" cholesterol level category.|True|
|10. The average BMI among patients with CVD and without CVD is different with a confidence interval of 95%|True|
|11. More than 50% of patients have hypertension (Grade 1 Hypertension or above)|True|
|12. More than 70% of patients with CVD have a CVD Risk Scale of "high" or above.|False|


### III. Multivariate Analysis

For the multivariate analysis, we used the following correlation methods:
1. Pearson Correlation Coefficient for continuous variables;
2. Point-Biserial Correlation between continuous variables vs the binary target variable;
3. Cramér's V Correlation between categorical variables vs the binary target variable.

Pearson and Point-Biserial Correlations are shown in the same heatmap, while Cramér's V Correlation will be displayed in a different one.

#### Continuous Variables vs Target Variable

![](img/heatmap1.PNG)

**Highlights**:
- A strong positive correlation between **weight** and **bmi**, **systolic (ap_hi)** and **diastolic (ap_lo)**;
- The target variable `cardio` has a mild positive correlation with **systolic (ap_hi)** and **diastolic (ap_lo)**, an almost zero correlation with **height**, and weak positive correlations with other variables;

#### Categorical Variables vs Target Variable

![](img/heatmap2.PNG)

**Highlights**:
- Strong positive correlation between **hyper** and **sys_diast**, **hyper** and **diastolic**,  **hyper** and **systolic**;
- The target variable `cardio` has a mild positive correlation with **systolic**, **diastolic**, **sys_diast**, and **hyper**. 
- The target variable `cardio` has a weak positive correlation with **bmi_class_enc**, **cholesterol**, and **cvd_risk**. 
- The target variable `cardio` has a zero or almost zero correlation with **gender**, **gluc**, **smoke**, **alco**, and **active_scale**. 

[back to top](#table-of-contents)

---

## 04. Data Preprocessing and Feature Selection
[(go to next section)](#05-machine-learning-modeling)

#### Data Preprocessing
In this section, the dataset was split into training, validation, and test. We did this for the convenience of preprocessing the columns in the right order, and also to avoid data leakage of training data into the other data.

Due to a profusion of outliers in numerical variables, we utilized the **RobustScaler** method on them. For categorical variables, we used the following criteria:

1. **Presence/absence of state**: gender, smoke, alco, active, hyper. Since these variables are binary ones indicating state, we will use **the One-hot Encoding**.
2. **Order of categories**: cholesterol, gluc, systolic, diastolic, sys_diast, bmi_class_enc, cvd_risk. Here we could have used an encoder that captures the frequency relationship between classes; however, each class has relevancy in terms of the order. Therefore, **the Ordinal Encoding method** was used.

#### Feature Selection

We have utilized three feature selectors to select the best features for our model:

1. **Boruta**: this wrapper method-based algorithm has selected only two variables: `ap_hi` (systolic values) and `bmi`. Since many relevant variables weren't selected, we won't use Boruta suggestion.
2. **Recursive Feature Elimination (RFE)**: Features `age`, `height`, `weight`, `ap_hi` were selected by the RFE. Since only `ap_hi` appeared in both RFE and Boruta, and many other relevant variables were excluded by both feature selectors, their recommendation won't be followed.
3. **Feature Importance**: we used the Logistic Regression, Random Forest Classifier, CatBoost Classifier, XGB Classifier, and the LGBM Classifier to generate the importance of each feature on each respective model:

- **Logistic Regression**: `alco_0`, `active_0`, `cvd_risk`, `ap_lo`, `smoke_0`,`bmi_class_enc`, `gender_1` have almost zero coefficient. Many variables are closer to zero coefficient, so it seems that each variable contributes little to explain the target variable.
- **Random Forest Classifier & Balanced Random Forest Classifier**: `alco_0`, `alco_1`, `smoke_0`, `smoke_1`, `active_0`, `active_1`,  `gender_1`,  `gender_2`, 
 `gluc`, `diastolic`, `sys_diast`,  `bmi_class_enc` have score values very close to zero.
- **CatBoost Classifier**:  `smoke_0`, `smoke_1`,`gender_1`,  `gender_2`,`bmi_class_enc`,`alco_0`,`alco_1`,`diastolic`,`hyper_0`,`hyper_1`,`active_0`, `active_1`, have scores values very close to zero.
- **XGB Classifier**:`diastolic`,`systolic`,`hyper_1`,`alco_1`, `active_1`,`bmi_class_enc`,`smoke_1`,`gender_2`,  have scores values very close to zero.
- **LGBM Classifier**:`diastolic`,`systolic`,`hyper_0`,`hyper_1`,`alco_1`, `active_1`,`bmi_class_enc`,`smoke_1`,`gender_2`,  have scores values very close to zero.

![](img/prep.PNG)
![](img/prep2.PNG)

According to the multivariate analysis and the features selected by the algorithms, we would drop the following variables:

- `bmi_class_enc`
- `gender_1`
- `gender_2`
- `smoke_0`
- `smoke_1`
- `alco_0`
- `alco_1`
- `active_0`
- `active_1`

However, we won't drop these variables since the medical literature has shown that there is a correlation between CVDs and these features. In a real situation, we would seek more data to get robust results. Therefore, we will proceed as it is.

[back to top](#table-of-contents)

---
## 05. Machine Learning Modeling
[(go to next section)](#06-hyperparameter-tuning)

This section was divided into three tasks:
  
* **Performance Metrics**: choose suitable metrics to measure the performance of the predictive model;
* **Modeling:** choose machine learning models to train and generate predictions
* **Cross-validation:** assess the trained models' real performance by testing the model within 10 folds.

### I. Performance Metrics

Metrics for a classification project work differently from regression ones. In regression problems, we are most concerned about reducing the error between the actual values and the predicted values, whereas in classification the concern is on maximizing the accuracy of the model. 

The term "accuracy" should be taken carefully, however. In classification, there is a myriad of metrics that measure the rate of right "guesses" against wrong ones; and it turns out that accuracy is just one of those metrics.

For problems in which the target variable has two labels (yes/no, positive/negative), **a confusion matrix** helps us calculating these metrics:

![](img/confusion.png)

_Image retrieved from [Towards Data Science.](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)_

On the top-side of the matrix, we have the actual values, and on the right side, we have the predicted values generated from the model. In the squares, there are four variables:

- **True Positive (TP)**: The number of observations that the model correctly predicted to be positive;
- **False Positive (FP)**: The number of observations that the model predicted to be positive when the actual value was actually negative;
- **True Negative (TN)**: The number of observations that the model correctly predicted to be negative;
- **False Negative (FN)**: The number of observations that the model predicted to be negative when the actual value was actually positive;

In most situations, the desired result would be to have zeroed in the FN and FP squares. This would indicate that the model accurately predicted the positive values as positive; and the negative values as negative.

There are lots of metrics to measure a model's capability of predicting TPs and TNs. A few of them will be explored now.

### Accuracy

![](img/accuracy.PNG)

**Accuracy is the number of correct predictions made divided by the total number of predictions made**, multiplied by 100 to turn it into a percentage.

Accuracy is a baseline metric to evaluate the performance of machine learning models. However, if your data has imbalance problems between classes, accuracy won't tell the whole story and will be not useful to measure performance. 

For example, let's say that model X predicts whether a person is sick (1) or not (0). If the data that model X uses for training contains 90% of sick individuals, the model will be biased towards predicting unseen data as sick.

Now let's say that 900 people are with the label "sick" (1) and 100 are labeled "not sick" (0). Suppose we train the model X on these past data, and we have generated predictions for 1000 observations coming from unseen data with the following results:

TP = 900
FP = 70
TN = 30
FN = 0

This model has accurately predicted all positives, while it poorly predicted the negatives (only 30/100 = 30% of correct predictions).

If one uses the accuracy formula presented above, we would get:

![](img/accuracy_1.PNG)

An accuracy of 93% seems nice, but we know that the model didn't label non-sick patients accurately.

### Precision

![](img/precision.PNG)

Precision measures **how well a model predicts positive labels**. Precision is a good measure when the costs of False Positive (say, patients without CVD but diagnosed with CVD) is high.

In the example we set, we would have a precision rate of:

![](img/precision1.PNG)

which basically says that the model has a 92% precision (or that the model predicted positive labels with an accuracy of 92%). 

In this project, we are dealing with a CVD detection challenge where the business earns $500 for each 5% increase in precision in diagnostics. Therefore, improving precision is one of the key goals for the business.

### Recall

![](img/recall.PNG)

Also known as sensitivity (or true positive rate), recall calculates **how many of the True Positives the model capture among all positives** in the dataset. Recall is a good measure when there is a high cost associated with False Negatives (in our case, patients with CVD but diagnosed without CVD).

Applying the recall formula to our example:

![](img/recall1.PNG)

As we expected, our model captures all true positives (100%).

We can also calculate the inverse of recall, which is the **specificity**. It measures **how many of the True Negatives the model capture among all negatives**:

![](img/specificity.PNG)

Applying the specificity formula:

![](img/specificity1.PNG)

which tells us that the model captures only 30% of all true negatives.

In this project, it will always be desirable to have diagnostics with a high recall rate from the patients' standpoint. The reason is simple: a low recall rate would be very costly for patients who have CVD but are mistakenly diagnosed without CVD.

### F1-Score

![](img/f1score.PNG)

The F-measure or balanced F-score (F1 score) is the harmonic mean of precision and recall. **F1 Score is needed when you want to seek a balance between Precision and Recall.** 

For reasons already discussed in the precision and recall sections, we will use the F1-Score as one of the key metrics in this project. 

### Balanced Accuracy

A more reliable option to accuracy is balanced accuracy. It is the average between the sensitivity (recall, or true positive rate) and specificity (the recall for negatives, or true negative rate):

![](img/balanced.PNG)

In the case of sick patients, we would have the following result:

![](img/balanced1.PNG)

which is a more reliable metric than the 93% accuracy we obtained earlier.

### AUC-ROC Score

![](img/ROC1.PNG)

**The Area Under the Curve - Receiver Operating Characteristics (AUC-ROC) score** is an important metric to evaluate a model's performance for classification problems. 

To understand the AUC-ROC score, we will explore the ROC curve above. The x-axis is the True Positive Rate (or Recall/Sensitivity), and the y-axis is the False Positive Rate (or 1 - Specificity). The ROC curve is the curve indicated by the 'ROC' in blue letters, and the AUC is literally the area under the ROC curve. **This area is the AUC-ROC score**; therefore, since the range of both axes is 1, we are talking about a squared area that can take the maximum value of 1 and a minimum value of 0.

A quick definition of TPR and FPR:

- **True Positive Rate** is the recall (or sensitivity). As we discussed earlier, it is the proportion of patients correctly classified as sick among all patients with a sickness.
- **False Positive Rate** is the proportion of patients incorrectly classified as sick (False Positives - FP) among all patients who are not sick (FP+TN). It can also be calculated by subtracting the specificity from 1.

To get the TPR and FPR values in our sick-or-not-sick example, we need to train a model and generate probabilities of patients being sick. Suppose we have a trained model and we want to test the model against five patients that we know their true health condition:

![](img/ROC3.PNG)

Then we generate probability predictions for these patients by using the trained model:

![](img/ROC4.PNG)

Now we need to set a probability threshold to determine whether the patient is sick or not. Let's check for a threshold of 0.50. If a patient has a predicted probability higher than 0.50, then it will be labeled as sick; otherwise, labeled as not sick:

![](img/ROC5.PNG)

With a 0.50 threshold, we have 2 patients who are sick, and 3 not sick. However, the model predicted one patient incorrectly. The confusion matrix shows the final outcome:

![](img/ROC6.PNG)

Now let's calculate the TPR and FPR for this specific threshold:

![](img/tpr.PNG)

Then we plot the resulted values in the AUC-ROC plot:

![](img/ROC7.PNG)

Okay, we got one point in our plot. Seems like the model with threshold 0.50 is very good in avoiding false positives (FPR = 0), but not perfectly good in predicting true positives (TPR = 0.66).

So what if we try **different thresholds** other than 0.50? Say, 0.10, 0.20, 0.30.... 0.90? Then we would get new TPs, TNs, FPs, TNs, and new TPR and FPR values. At the end of the day, we would have many points in our graph:

![](img/ROC8.PNG)

It turns out that **the union of these points is the ROC curve**, and **the area under the curve is the AUC-ROC score** for the trained model utilized in this example:

![](img/ROC1.PNG)

**Ideally, we would have an AUC-ROC curve that represents the total area of a square (AUC-ROC score = 1)**. This means that, at a certain threshold, our model has a True Positive Rate of 1 (100% of patients correctly classified as positives), and a False Positive Rate of 0 (no false positives generated by the model). In our project, an AUC-ROC score of 1 means that the model predicts patients with CVD and without CVD correctly. In the plot, this excellent point is located where the y-axis corresponds to 1 and the x-axis is equal to 0:

![](img/ROC2.PNG)

The worst situation is when the AUC-ROC curve is 45º degrees, which means that every TPR has the same value as the FPR. For certain thresholds, the model is accurate to predict true positives but completely inaccurate in predicting negatives (TPR = FPR = 1), or the opposite: completely inaccurate predicting positives but accurate predicting negatives (TPR = FPR = 0). 

![](img/ROC9.PNG)

Another type of situation is when the AUC-ROC score (the area under the ROC curve) is zero. Then we see an unusual behavior: the model predicts all positives as negatives, and negatives as positives:

![](img/ROC10.PNG)

In summary, the ROC curve is a summary of all TPR/FPR outcomes that are generated by different threshold values by a model. 

The area under the curve (AUC) is also useful to compare the performance of different models, such as this one:

![](img/ROC11.PNG)

In this example, we are comparing the XGB Classifier model and the Logistic Regression. By looking at their ROC scores (the area under the curve), we can conclude that the XGB model is better than the Logistic Regression across all thresholds.

Credit: this AUC-ROC explanation was inspired by the amazing explanation made by [Josh Starmer from StatQuest.](https://www.youtube.com/watch?v=4jRBRDbJemM)

References:
- [Understanding AUC - ROC Curve, by Sarang Narkhede](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
- [How to Use ROC Curves and Precision-Recall Curves for Classification in Python, by Jason Brownlee](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/).

### Cohen-Kappa Score

The Cohen-Kappa score is a measure of interrater reliability. The metric calculates the reliability of votes given by two raters (say, rater A and rater B), who have the task to vote whether a particular sample (or patient, in our case) is sick or not:

![](img/kappascore.PNG)

where,

![](img/kappascore1.PNG)

Cohen-Kappa score values are always less than 1, being 1 equal to an agreement between both raters about the decision made, and 0 equal to agreement by chance. If the score is less than 0, it means that the model is generating predictions that are worse than a model that randomly assign labels "sick" or "not sick" to patients (agreement by chance). Thresholds were defined by Landis and Koch (1977): 

|Kappa value|Agreement level|
|--|--|
| < 0| agreement by chance|
| 0–0.20| slight agreement|
| 0.21–0.40| fair agreement|
| 0.41–0.60| moderate agreement|
| 0.61–0.80| substantial agreement|
| 0.81–1 | almost perfect agreement|
| 1 | perfect agreement |

Let's set an example: 

![](img/kappa.PNG)

In this example, we see a total of 10 votes where:
- Rater A assigned 5 patients as sick, and rater B agreed
- Rater A also assigned 2 patients as "not sick", and rater B agreed. 
- Rater A labeled 1 patient as sick, while rater B disagreed. 
- Rater A labeled 2 patients as "not sick", and rater B disagreed.

To calculate the probability of agreement between raters (Po), we sum the total of agreements (diagonal line) divided by the total votes:

![](img/kappascore2.PNG)

Now we calculate the probability of agreement between raters when each rater assigns labels randomly (Pe). To help us during the calculation, an updated table with totals is displayed below:

![](img/kappa1.PNG)

To get this probability, we calculate the following:

- **The probability that both raters agree on the label "sick" by chance**: probability of rater A assigning "sick" (6/10) **times** probability of rater B assigning "sick" (7/10).
- **The probability that both raters agree on the label "not sick" by chance**: probability of rater A assigning "not sick" (4/10) **times** probability of rater B assigning "not sick" (3/10);

Then we sum both probabilities:

![](img/kappascore3.PNG)

Now we can proceed to calculate the Cohen-Kappa score:

![](img/kappascore4.PNG)

The Cohen-Kappa score of 0.34 means that, when two raters use a trained model to assign labels, they are in a fair agreement.

References: 
- [The Data Scientist - Performance Measures: Cohen’s Kappa statistic](https://thedatascientist.com/performance-measures-cohens-kappa-statistic/)
- [Multi-Class Metrics Made Simple, Part III: the Kappa Score (aka Cohen’s Kappa Coefficient), by Boaz Shmueli](https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c)

### Brier Score Loss

The Brier Score Loss (or Brier Score) measures the mean squared error between predicted probabilities of an outcome (say, a patient has CVD) and the actual outcome. 

![](img/brierscore.PNG)

where,

![](img/brierscore1.PNG)

Given the nature of the formula, the brier score varies from 0(the best possible score) to 1 (the worst score). 

Let's set one example. Suppose that we have a model that predicts whether two patients have or not CVD. Their predicted probabilities of having CVD are:

P1 = 0.95 (patient 1 has 95% chance of having CVD)
P2 = 0 (patient 2 has 0% chance of having CVD)

Patient 1 is actually with a CVD, and patient 2 doesn't have CVD (the model was accurate). Translating it into the Brier Score formula, we have:

![](img/brierscore2.PNG)

which is a very low Brier Score, showing that the model makes good predictions.

It is worth mentioning that the Brier Score is an accuracy metric suitable for models predicting binary outcomes (yes/no, true/false, positive/negative). 

References:

- Application in python: [Scikit-learn's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html).

### II. Modeling 

In this project, 9 classifiers were trained with the training and validation datasets. We got the following results:

![](img/modeling.PNG)

From the table, the CatBoost Classifier, XGB Classifier, and LGBM Classifier are the three best performers in terms of F1-Score - which is the metric we are using as the primary criteria for choosing the best model for our project.

Before we proceed, let's explore each classifier and understand their mechanisms.

### 1. Random Guess (baseline)
The first model is a baseline. We randomly choose predicted values and compared them with the actual values. This is supposedly the worst model performer from our set of chosen models; therefore, if a particular model
 performs worse than the baseline, we would not use it as the chosen one.


### 2. Logistic Regression

When the target variable is of binary type (yes/no, positive/negative) and there is a linear relationship between independent variables and the target, a logistic regression model can be used to predict probabilities of an observation belonging to a certain label.

While a [linear regression](https://github.com/alanmaehara/Sales-Prediction#2-linear-regression) model tries to fit the best line by using the Ordinary Least Squares (OLS) method, logistic regression uses a technique called Maximum Likelihood Estimator (MLE). The reason is that, for dependent variables which allow for only two answers, and for independent variables who are linearly related to the dependent variable, fitting a straight line won't be the best choice: a curved, usually called sigmoidal curve, would be ideal:

![](img/logistic1.PNG)

In the example plotted above, we assume that the weight of patients is linear to the probability of having CVD. 

If we define a probability threshold of .50, all patients who got a probability of having CVD above .50 will be labeled as having CVD (1) and zero otherwise. 

At this point, one can guess why should logistic models be avoided if your data is not linearly related. Imagine if weight has nothing to do with a higher chance of getting CVD, and the dots in the plot were shifted randomly across the x-axis. Then the logistic model would hardly capture data patterns that would accurately predict probabilities of a patient having CVD.

Linear regression models have some similar fundamentals to logistic regression ones. For instance, modeling a linear regression model is somewhat similar to logistic regression: we have the coefficients for each independent variable, and we can calculate their values to check whether a specific variable explains better the phenomenon (dependent variable) than the others:

**Formula for linear regression (with one independent variable X1):**

![](img/lin.PNG)

where the betas are the coefficients, the output value y is the predicted value, and the X1 is the value of an independent variable (eg: if weight, it could assume values from 34 to 180 kg according to our data).

**Formula for logistic regression (with one independent variable X1):**

![](img/log.PNG)

Notice that there is a natural log term that represents the predicted value. **This is the natural log of the odds** (also called log-odds), in which _p_ is the probability of an individual having CVD and _1-p_ is the probability of not having CVD.

Why the natural log is utilized here? Let's take a look at the odds first. As an example, let's say that from 10 individuals weighing 80 kg, 8 have CVD, and 2 doesn't. The probability of a patient having CVD is 0.8 and the probability of not having CVD is 0.2. Then the odds of having CVD are:

![](img/log1.PNG)

Then let's calculate the odds of having CVD for patients weighing 50kg: 2 have CVD and 8 doesn't:

![](img/log2.PNG)

As one can tell, when the numerator is smaller than the denominator, odds range from 0 to 1. However, when the numerator is bigger than the denominator, odds range from 1 to infinity. This is the reason why logarithms are utilized in the logistic regression; **it is a measure to standardize output values across all probability values:**

![](img/log3.PNG)

With the natural log of the odds for these two specific cases on hand, we can figure out the probability of a patient with 50kg to have CVD and the probability of a patient with 80kg to have CVD. We do the following:

- Exponentiate the log-odds:

![](img/log4.PNG)

- Do some algebra towards isolating p:

![](img/log5.PNG)

- We can translate the p as:

![](img/log51.PNG)

If we plug the numbers we got earlier in the formula:

![](img/log6.PNG)

which tells us that the likelihood of a patient with 80kg having CVD is 80%, and the likelihood of a patient with 50kg having CVD is 20%. If we plot the probability predictions in the plot, we see that they are indeed part of a sigmoidal curve:

![](img/logistic2.PNG)

But how to tell that the curve is a good fit for the data? This is where Maximum Likelihood Estimation (MLE) comes into play. Suppose that we have calculated likelihoods (probabilities) for all kinds of patients in our dataset, and we have fitted a curve like the first one we saw here:

![](img/logistic1.PNG)

Now imagine that we got the following likelihoods of having CVD for patients with CVD (0.70, 0.80, 0.90, 0.95) and for patients without CVD (0.10, 0.20, 0.30, 0.40). Then we calculate the log-likelihood of the model:

- Log-likelihood of having CVD:

![](img/log7.PNG)

- Log-likelihood of not having CVD:

![](img/log8.PNG)

- Summing up both log-likelihood: 

![](img/log9.PNG)

With the current parameters, the logistic model has a log-likelihood of -1.90. Perhaps there are other parameters that could increase the log-likelihood value (the higher, the better), but this is a task that is efficiently done by any statistical software. 

In practice, logistic models can take more than one independent variable, which makes the model even more complex than the one we depicted here. 

Let's summarize the good and bad aspects of the Logistic Regression classifier:

**The good:**
- Like linear regression, the logistic regression is a neat way to describe the effect of each independent variable on the target;
- Algorithm implementation is simple and fast, training is efficient;
- Overfitting can be avoided by using penalty techniques such as the L1 and L2.

**The bad:**
- Only works when the target variable is binary;
- Assumes linearity between independent variables and the target;
- Multicollinearity between independent variables is not permitted;
- It doesn't model non-linear data well.

Additional sources to build intuition:
- [Brandon Foltz's](https://www.youtube.com/watch?v=zAULhNrnuL4) series on Logistic Regression
- [Josh Starmer's](https://www.youtube.com/watch?v=vN5cNN2-HWE) videos on the topic
- [Application in python - Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic#sklearn.linear_model.LogisticRegression)

### 3. K-Nearest Neighbors (KNN) Classifier

The KNN Classifier is one of the simplest machine learning algorithms for classification. To classify an unknown data point to a certain class, it maps the K-nearest neighbors from that data point, analyzes the labels of the K-data points selected, and the majority number of labels will determine the unknown data point label.

For example: by choosing K=3, we detect that 2 patients labeled as "with CVD" and 1 patient labeled as "without CVD" are the nearest 5 neighbors from the unlabeled (unknown) patient. As the majority of the neighbors are with CVD, the unknown patient will be labeled as with CVD.

The following example uses only two features (height, weight) to plot the data points, but in practice, this could be n-features.

![](img/knn.PNG)

For K = 5, we would have a different result:

![](img/knn1.PNG)

To calculate the K-nearest neighbors, we calculate the distance (in most cases, the euclidean distance) between the unlabeled point and each labeled point. The K neighbors with minimum distance value are the K-nearest neighbors.

Now let's try to calculate the euclidean distance of the unlabeled point and all other labeled points. Suppose that we have the following dataset that resembles the graphs shown above:

![](img/knn2.PNG)

The Euclidean distance is defined as:

![](img/knn4.PNG)

In the example given, we would do like this for just two features (weight and height: n = 2). Features are represented by the letter i:

![](img/knn5.PNG)

Therefore, the euclidean distance between patient 1 and patient 2 is 5. To speed up to a conclusion, distances between patient 1 and other patients are displayed below:

![](img/knn3.PNG)

If we picked K = 3, then we would choose patients 2, 3, and 4 as the 3-nearest neighbors to patient 1. Since patients 3 and 4 are with CVD, we would classify patient 1 as with CVD.

The optimal K number is defined in a stochastical way, although an odd-numbered K is recommended if the number of classes are even.

Let's summarize the good and bad aspects of the KNN classifier:

**The good:**
- No assumption on the data distribution is taken (non-parametric);
- Lazy algorithm: training is fast, since it does computation only for the testing phase;
- KNN usually performs fairly well when data has a small number of features
- Effective for large training datasets;
- Simple implementation.

**The bad:**

- Low values for k can generate noise and get affected by outliers;
- High values for k are not optimal when data has a few samples;
- Being a lazy learner algorithm means that the testing phase is slower;
- Due to the curse of dimensionality, the algorithm performance can be negatively affected when there are many features in the dataset;
- As K increases, so does computation cost.

Additional sources to build intuition:

- [KNN (K-Nearest Neighbors) #1 - Italo José (in Portuguese)](https://medium.com/brasil-ai/knn-k-nearest-neighbors-1-e140c82e9c4e)
- [KNN Classification using Scikit-learn - Datacamp](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)
- [Application in python - Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

### 4. Naive Bayes

The Naive Bayes classifier is a probabilistic classifier based on the [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) that has two main assumptions: (1) features are independent of each other; (2) all features have equal importance (equal weight) to explain the target variable.

There are two main types of Naive Bayes classifiers: (1) the Multinomial Naive Bayes, which works best for categorical features; (2) the Gaussian Naive Bayes, which works for both categorical and numerical features under the assumption that all features belong to a normal (gaussian) distribution. To build intuition on the topic, we will focus on explaining the Multinomial Naive Bayes.

First, imagine that we have eight patients with CVD, four patients without CVD, one patient whose health condition is unknown and we want to calculate the probability of having CVD for that particular patient. Features we would consider are four: alcoholic consumption, obesity, diabetes, hypertension. All of them indicate presence/absence (yes/no).

![](img/naive.PNG)

Let's use Naive Bayes to predict the probability of this unlabeled patient (P1) having CVD based on the information we know from the eight patients. The Naive Bayes formula is as follows:

![](img/nb.PNG)

in which we can translate as:

![](img/nb1.PNG)

where,

![](img/nb2.PNG)

Let's start by calculating the conditional probabilities. To better calculate them, we summarize the table we saw earlier in confusion matrices per feature:

![](img/naive1.PNG)

Then, we calculate two probabilities: (1) the probability of patient 1 (P1) of having CVD given his own set of features; (2) the probability of patient 1 of not having CVD. Remember: patient 1 only has hypertension:

![](img/nb3.PNG)

Since the denominator term is equal for both, we can get rid of it. Then we get the following numbers:

![](img/nb4.PNG)

If one sums up both probabilities, they should be 1. Let's normalize them:

![](img/nb5.PNG)

Since the probability of patient 1 having CVD given his features X is higher than the probability of not having CVD, we classify patient 1 as having CVD.

Let's summarize the good and bad aspects of the Naive Bayes Classifier:

**The good:**
- Very fast training compared to other classifiers;
- Can be modeled with a small training dataset; 

**The bad:**
- Assumes that variables are independent of each other. Therefore, it is not a good algorithm if there are dependent features in the dataset;
- Assumes that all variables have the same weight to determine the outcome;
- If assumptions are not met, probability outputs are not reliable (hence the name "naive");
- Testing phase: if an unknown class appears in the test dataset, a zero probability will be given. To avoid this, the Laplace estimation needs to be performed;

Additional sources to build intuition:

- For a similar example, check [geekforgeeks.org - Naive Bayes Classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/);
- For Gaussian Naive Bayes, check [StatQuest by Josh Starmer](https://www.youtube.com/watch?v=H3EjCKtlVog);

### 5. Stochastic Gradient Descent (SGD) Classifier

In optimization problems for Machine Learning, we are usually worried about reducing the error or increasing the accuracy of our predictions when compared to the observed values. [Linear Regression](https://github.com/alanmaehara/Sales-Prediction#2-linear-regression) models perhaps provide the clearest example of how we try to fit a line that reduces such error to the minimum value possible:

![](img/mpe.png)

_A linear regression line in blue with observed values in black, and errors in red. Graph retrieved from [STHDA](http://www.sthda.com/english/articles/40-regression-analysis/167-simple-linear-regression-in-r/)._

In linear regression models, the best line is fit in a way that minimizes the error through a loss function. For example, the residual sum of the squares is a loss function where the goal is to find parameter values that minimize the residual. For a single linear regression model, the parameters are the slope (m) and the intercept (b) of the function (recall linear functions like y = mx + b).

Stochastic Gradient Descent also has the same goal of deriving slopes and intercepts towards reducing the discrepancy between the observed value and the predicted value given by the model. It does so by using a loss function like the linear regression model, but in a slightly different way and faster than the traditional Gradient Descent algorithm. Let's delve ourselves first into the fundamentals of the Gradient Descent, and later we find out its differences from the Stochastic Gradient Descent.

To build intuition, let's say that we have three people who want to predict their heights, and we wish to do so by fitting the best prediction line possible. We will set this line by tentative, with a slope of 1 and a y-intercept of 0 (a 45º degree line):

![](img/sgd.PNG)

As one can tell, the model's prediction line is far from being optimal. The error is just too big. See the exact numbers in the table below:

![](img/sgd1.PNG)

But how can we reduce the errors? For Gradient Descent models, we set a loss function and work towards giving "small", gradient steps towards the optimal line's intercept and slope values.

The loss function can be any; in this example, we use the residual sum of the squares:

![](img/sgd3.PNG)

where,

![](img/sgd4.PNG)

Let's calculate the residual sum of squares (RSS):

![](img/sgd5.PNG)

To find the optimal values for the intercept and the slope, we take partial derivatives of the RSS with respect to slope and height:

![](img/sgd6.PNG)

At this point, one might say to just set the derivatives to zero to find the optimal slope and intercept. However, there are some cases that we can't derive functions and set them to zero. In either case, the gradient descent works just fine since it approaches the optimal slope and intercepts to zero by taking small steps.

Now, we guess some numbers for the intercept and the slope to calculate the RSS and its derivatives. Let's try slope 1 and intercept 0, just like our initial graph:

![](img/sgd7.PNG)

We got an RSS of 2.18, and negatives slopes. Since our goal is to get slopes closer to zero (which would imply a lower RSS), the gradient descent will calculate a new intercept and slope based on the results we got. We just need to calculate the step size for intercept and slope based on a learning rate (in this example, we set to 0.01), and then subtract the step size by the previous intercept:

![](img/sgd8.PNG)

Note: the higher the learning rate, the higher the step size becomes. Ideally, one would set a lower learning rate so that intercept and slope values increase gradually.

With the new intercept and slope values, we calculate the loss function and the derivatives again:

![](img/sgd9.PNG)

In this iteration, the gradient descent gave a step towards reducing the RSS. We would continue until we literally get step sizes closer to zero (if one continues iterating, the learning rate would reduce the slope and intercept closer to zero).

A problem that the Gradient Descent algorithm has is in the computational aspect of it. If a dataset has more than just weight and height as independent variables (say, eight) with thousands of data points, then the algorithm would have to calculate derivatives for every single independent variable with thousands of terms on each derivative. This is clearly expensive when it comes to computational time, and this is where the Stochastic Gradient Descent (SDG) comes into play. The SDG differs from the Gradient Descent by selecting random samples from the dataset (or picking small subsets of data) rather than using all data points to calculate the gradient descent.

The SDG is not a model by itself; it is rather a way of optimizing an existing model such as the Logistic Regression, Support Vector Machine, or the Linear Regression.

According to scikit-learn API, the SDG has the following pros and cons: 

**The Good**:

- Overall efficient, especially if compared to traditional (Batch) Gradient Descent;
- Implementation is straightforward and many parameters can be tuned.

**The Bad**:

- SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations;
- SGD is sensitive to feature scaling;

Additional sources to build intuition:

- For scikit-learn documentation on the SDG algorithm, check [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html);
- For building visual intuition, check [StatQuest by Josh Starmer](https://www.youtube.com/watch?v=vMh0zPT0tLI).

### 6. Random Forest and Balanced Random Forest

Random forest for classification problems has a similar mechanism intuition to random forest for regression. I have derived proper explanations on decision trees (which is the basis for random forests) and the random forest algorithm itself in a sales prediction project located [here](https://github.com/alanmaehara/Sales-Prediction#decision-trees), so I will just highlight the main differences between random forests and balanced random forests.

* **Balanced Random Forest vs Random Forest**

The Balanced Random forest algorithm differs from the traditional random forest by the fact that the former deals with imbalanced classes. For example, if the target variable of a dataset has a strong imbalance between classes, it would be preferable  to use a balancing technique before preprocessing the dataset - or train a model that handles imbalancing like the Balanced Random Forest.

The algorithm is provided by [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and the class we utilized for this project is the `BalancedRandomForestClassifier`. This specific algorithm randomly undersamples the majority class till it gets to the same number of samples for the minority class. The technique used is bootstrapping, which is just a fancy name for randomly sample data points with replacement.


Random Forest (in general) have the following pros and cons:

**The Good**:

- Usually performs well to most datasets;
- Due to its random nature, it performs better than decision trees;
- Intelligibility and easiness to interpret;
- Default parameter values usually perform well, which makes implementation easier;
- Like linear models, it selects the most predictive features to generate predictions and somehow serves as a feature selector method;

**The Bad**:

- Training is slow and parameter tuning is necessary to improve training times;

Additional sources to build intuition:

- For scikit-learn documentation, check [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html);

### 7. XGB Classifier

The XGB algorithm also works for both classification and regression problems. I have also written explanations [here](https://github.com/alanmaehara/Sales-Prediction#7-xgboost-regression) on the Gradient Boost algorithm (which is the basis for the XGB model along with random forests) and the XGB Regressor algorithm, which has similar ideas with the XGB Classifier. 

As in gradient boosting, XGB models use the principle of ensemble learning for training. Ensemble learning focuses on two things:

- **Bagging**: a bootstrapping step (similar to what happens to random forests) where a sample of row (data points) and columns (features) are selected at random to grow a tree. This is an iterative procedure for n trees, where at the end of the procedure an average (for regression) or vote majority (for classification) is calculated among all n trees to reach a final prediction for an unlabeled data point. This is very useful to reduce variance across the model;
- **Boosting**: as the model generates predictions, the boosting step works towards creating new models that reduce the errors of previous models in a sequential way. The method to reduce the errors comes from a mathematics problem: to minimize a [cost function](https://en.wikipedia.org/wiki/Loss_function) by using the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent).

The difference between Gradient Boosting and XGB model is computational: the XGB focuses on providing the gradient boosting a more optimized way (in terms of time and overfitting control) of training. More details can be found in the [documentation](https://xgboost.readthedocs.io/en/latest/).

As usual, let's highlight the pros and cons of XGB:

**The Good**:

- The XGB has a built-in regularization parameter that lessens the effect of unimportant features during prediction. 
- XGB also imposes a regularization on each tree's leaves, which helps reducing overfitting;
- Tree Pruning: If necessary, XGB prune leaves (and even trees), which helps to avoid overfitting.
- Computation Processing: XGB trains faster than gradient boosting and some other models due to its parallel processing
- In-built Cross-Validation: the algorithm can handle cross-validation for each iteration, and use the optimum number of iterations according to prediction results;
- It deals with missing values by default;

**The Bad**:

- Parameters are vast, and to unlock all the potential in XGBoost models, one might have to optimize parameters. A nice guide for parameters is [Aarshay Jain's](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/) article on parameter tuning;

Additional sources to build intuition:

- For documentation, check [here](https://xgboost.readthedocs.io/en/latest/). It also contains detailed information on the theory behind gradient boosting models and the XGB.


### 8. Support Vector Machines (SVM)

Support Vector Machines (SVM) are one of the most popular machine learning algorithms for classification and regression problems. The SVM tries to divide a sample space in a way that separates classes according to a maximum possible margin. 

Let's say we are trying to fit two SVM lines (called Support Vector Classifier - SVC) to separate patients without CVD (negative, red) and with CVD (positive, green) as above:

![](img/svm.PNG)

Which SVC would be the best one to split the classes? If one sees the margin size on each graph (calculated according to the nearest patients to the SVM lines in both classes), one would conclude that the left-side SVM1 line separates classes better:

![](img/svm1.PNG)

In practice, the SVM algorithm handles calculation for us towards finding the line with the largest margin possible. This would ensure that our predictions are more reliable. For example, look at the graphs depicted below with a new patient added inside the margin:

![](img/svm3.PNG)

If one uses the SVC1 model to predict the health condition of this new patient, it would be labeled as positive (with CVD). It makes sense since such a patient has a weight that is closer to the cluster of patients with CVD. However, if one uses the SVM2 model, then the patient would be labeled as without CVD.

How about non-linear data? How SVMs fit an SVC so that it can separate between classes? In this case, SVMs have something called **SVM kernels**, in which kernels transform a particular data to higher dimensions to facilitate the separation of classes. There are three types of kernels: linear, polynomial, and radial basis function. 

For example, let's say that we have the following dataset:

![](img/svm4.PNG)

Then, we can use an SVM's polynomial kernel that uses the y-axis (height) squared. The same graph would look like this:

![](img/svm5.PNG)

Although the best margin is quite short, the SVM could fit an SVC such that it distinguishes datapoints to distinct classes.

The mathematical intuition is well covered by [MIT's Profº Patrick Winston](https://www.youtube.com/watch?v=_PwhiWxHK8o), so we won't cover it here.

According to Scikit-learn, the pros and cons of SVMs are:

**The Good**:

- Effective in high dimensional spaces;
- Effective in cases where the number of dimensions are greater than the number of samples;
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient;
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.


**The Bad**:

- The algorithm doesn't have a method that returns probabilities; it just displays the predicted label;
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
- When data is non-linear, computation costs can get very expensive; in this case, use the Stochastic Gradient Descent as a method to optimize the SVM. 

Additional sources:
- For application in Python, check [Scikit-learn](https://scikit-learn.org/stable/modules/svm.html)
- [Support Vector Machine — Introduction to Machine Learning Algorithms](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47) by Rohith Gandhi.

### 10. CatBoost Classifier

In order to train machine learning classifiers, one is perhaps already used to the idea of encoding categorical variables that contains text (strings) to numerical values since most algorithms can't handle categorical variables by default. However, the CatBoost Classifier is an exception and its own name suggests what it does: "Category" and "Boosting".

If one is seeking an algorithm that not only deals with categorical variables in an excellent way but also leverages all the benefits that a gradient boosting model has to offer, the CatBoost Classifier is certainly an attractive option. 

**The Good**:

- Similar performance with the most state-of-the-art machine learning algorithms;
- Leverages all gradient boosting benefits;
- Handle categorical data by using one-hot encoding;
- Default parameters perform quite well.


**The Bad**:

- If compared to other "boosting" models, it can take more training time (see [Tal Peretz's](https://towardsdatascience.com/https-medium-com-talperetz24-mastering-the-new-generation-of-gradient-boosting-db04062a7ea2) article; he does a performance check with CatBoost vs XGBoost).
- If categorical variables aren't present in the model, it might have lower performance when compared to other boosting models.

Additional sources:
- For documentation and tutorials, check [Catboost's documentation](https://catboost.ai/docs/concepts/tutorials.html).

### 11. LightGBM (LGBM) Classifier

The LightGBM (aka "Light Gradient Boosting Model") is also a gradient boosting-based algorithm with a focus on fast training times while improving accuracy rates. 

In general, the main benefits of using the LGBM are:

1. LGBM uses histograms to store continuous values into bins. This mean reduced calculating times for each node split on trees and less memory usage;
2. While XGBoost and other decision tree-based algorithms grow trees by level (depth)-wise like this:

![](img/lgbm.png)

LGBM grows trees leaf-wise:

![](img/lgbm1.png)

which helps increasing accuracy, since it splits the leaf with the best potential to reduce loss (the difference between predicted and actual values).

The LGBM has also some advantages over decision tree-based algorithms such as categorical feature optimization and parallel learning optimization. For more details, check the [LGBM documentation](https://lightgbm.readthedocs.io/en/latest/Features.html).

**The Good**:

According to the [documentation](https://lightgbm.readthedocs.io/en/latest/), the following advantages are observed when comparing the LGBM to other decision tree-based algorithms:
- Faster training times and higher efficiency;
- Low memory usage;
- Better accuracy;
- Support of parallel and GPU learning;
- Capable of handling large-scale data;
- Deals with unprocessed categorical variables.

**The Bad**:

- With a huge amount of data, some parameter tweaks might be necessary to start training.


Additional sources:
- A great comparison between CatBoost, XGB, and LGBM models by [Alvira Swalin](https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html);
- For documentation, check [here](https://lightgbm.readthedocs.io/en/latest/).

### III. Cross-Validation

In this step, cross-validation was performed on CatBoost Classifier, XGB Classifier, and LGBM Classifier using 10 folds. The training and validation data were combined together in this step and the results are displayed below:

![](img/cv.PNG)

By a very short margin, the LGBM Classifier was the best performer with an F1-Score of 0.7250. Combined with the results we saw in the previous section, we will utilize the LGBM Classifier for this project.

[back to top](#table-of-contents)

---
## 06. Hyperparameter Tuning

[(go to next section)](#07-business-performance)

In this project, we will use the Random Search method on the LGBM Classifier. The LGBM Classifier has the following parameters:

1. `max_depth`: maximum depth of a tree (or maximum number of nodes). Deeper trees can model more complex relationships by adding more nodes, but as we go deeper, splits become less relevant and are sometimes only due to noise, causing the model to overfit. Tree still grows leaf-wise;
2. `num_leaves`: max number of leaves in one tree;
3. `min_data_in_leaf`: minimal number of data in one leaf. Can be used to deal with over-fitting;
4. `learning_rate`: Also known as "eta". A lower eta makes our model more robust to overfitting thus, usually, the lower the learning rate, the best. But with a lower eta, we need more boosting rounds, which takes more time to train, sometimes for only marginal improvements;
5. `colsample_bytree`: ratio from 0 to 1 representing the number of columns used by each tree. Can be used to speed up training and deal with overfitting with a low number; for example, if the chosen value is 0.90, the algorithm will select 90% of features at each tree node;
6. `n_estimators`: number of boosting iterations;
7. `min_child_weight`: is the minimum weight (or number of samples if all samples have a weight of 1) required in order to create a new node in the tree. A smaller `min_child_weight` allows the algorithm to create children that correspond to fewer samples, thus allowing for more complex trees, but again, more likely to overfit;

Using the optimal set of parameters, we obtained the following results:

![](img/lgbm_tuned.PNG)

Overall, we observe a 1% increase in F1-score after tuning. The recall drastically increased to ~7% at the cost of reducing precision by 5%.

* **Calibration**

Some classification algorithms might be uncalibrated when generating probabilities of a specific label. For example, calibrated models that predict (for a certain number of patients) a probability of 20% of having CVD means that approximately 20% of such patients actually have CVD. When such numbers are too sparse, we have an uncalibrated model that needs calibration in order to predict more reliable results. To perform calibration, a regressor is utilized to map the probability outputs to the true probability; for more details, check documentation in [scikit-learn](https://scikit-learn.org/stable/modules/calibration.html#calibration).

Some linear models such as the Logistic Regression already produce calibrated probabilities whereas some others don't. In this project, we analyzed whether calibration would be necessary for the LGBM model, so we compared the LGBM model performance before and after the calibration procedure was held with test data.

To interpret the calibration curve:
- **y-axis** = actual probability of a positive event for a certain sample;  
- **x-axis** = predicted probability of a positive event for a certain sample; 
- The closer the curve is to the perfectly calibrated line (45 degrees), the more calibrated the model is:

![](img/lgbm_calibration.PNG)

As we can observe, the tuned LGBM model presents a poor performance with probabilities around 0.40~0.60. The LGBM model with default parameters (in blue) is surprisingly more stable, along with the tuned LGBM model with isotonic regressor. We ran cross-validation with the two best performers (LGBM and the calibrated model)

![](img/final_result.PNG)

Differences in F1-Scores are minimum. To keep things simple, we opted to use the LGBM model without tuned parameters and without calibration as the main model.

[back to top](#table-of-contents)

---
## 07. Business Performance
[(go to next section)](#conclusion)

The model we chose generates CVD predictions with precision between **76.03% and 75.03%**. Since there are more than 70,000 patients in the dataset, we can calculate the firm's current profit based on the current solution (healthcare software) performance, and compare it with the machine learning solution we have built for this project. 

Important notes:
- For each 5\% increase in precision rate above 50\%, **there is a \$500 increase for the patients' bill**;
- For unitary percent increase, the patient's bill increase accordingly ($100);
- The diagnosis cost is **\$1000 per patient**;
- For the existing solution, the precision rate varies from **55% (worst scenario) and 65% (best scenario)**;
- For the LGBM model, the precision rate varies from **75.03% (worst scenario) and %76.03 (best scenario)**

Total profit was:

![](img/business.PNG)

For 70,000 patients, **the current operation (software solution)** would have a debt of ~\$35 million in the worst scenario, and would have a profit of ~\$35 million in the best scenario.

**Under the model built in this project**, the firm would never see a negative value: in the worst scenario, the profit is around \$105,2 million; in the best scenario, profit would be \$112,2. million. 

Considering the best scenario possible, **our model would increase revenues to the order of 68\%** compared with the current software solution. 

[back to top](#table-of-contents)


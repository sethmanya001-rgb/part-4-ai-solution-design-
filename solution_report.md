Part 4: AI Solution Design for a Business Problem
Task 1: Business Domain
Selected Domain: Healthcare
Healthcare is one of the most impactful domains for AI. Early and accurate detection of diseases can save millions of lives. Hospitals and clinics generate massive amounts of patient data daily including medical records, lab reports, symptoms, and diagnostic images, most of which goes underutilized. AI can analyze this data to assist doctors in making faster and more accurate decisions.
Task 2: Define the Business Problem
Problem Statement: Early Detection of Diabetes in Patients using AI
Diabetes affects over 537 million adults worldwide. The majority of cases are detected only after serious complications arise, by which point the disease has already caused significant damage. Early detection can prevent these complications and reduce treatment costs by up to 80 percent.
Stakeholders:
Patients are the primary beneficiaries who get early warnings.
Doctors and clinicians use AI predictions to support diagnosis.
Hospital management benefits from reduced costs and improved patient outcomes.
Insurance companies get better risk assessment and pricing.
Government health departments can monitor disease at the population level.
Current Manual Process:
Patients visit a doctor only when they feel sick. The doctor orders blood tests like HbA1c and fasting glucose. Lab results take 1 to 3 days. The doctor manually reviews results and makes a diagnosis. High-risk patients are often not identified until complications arise.
Limitations of Current Process:
The process is reactive, not proactive, so patients are only diagnosed after symptoms appear. It is time-consuming because waiting for lab results delays treatment. Human error means doctors may miss early warning signs in large volumes of data. Frequent manual testing is expensive for patients. Doctors cannot easily spot combinations of subtle risk factors across thousands of patients. Rural areas lack access to specialist doctors.
Task 3: AI Task Type
Selected AI Task: Classification (Binary)
The output we need is whether a patient is at risk of diabetes or not. This is a binary classification problem where 1 means at risk and 0 means not at risk. Patient features like age, BMI, glucose level, and blood pressure are used as inputs. The model learns patterns from historical patient data where outcomes are already known. Classification is the most suitable task because the target is a discrete label, historical labeled data is available, and the output needs to be interpretable by doctors.
Task 4: Data Requirement Plan
Type of Data: Structured tabular patient data from hospital records and health checkups.
Input Features:
Age is the patient's age in years.
BMI is the Body Mass Index.
Blood Glucose Level is fasting blood sugar in mg per dL.
HbA1c Level is average blood sugar over 3 months.
Blood Pressure is systolic blood pressure in mmHg.
Insulin Level is insulin in blood.
Skin Thickness is triceps skinfold thickness.
Family History is 1 if a family member has diabetes.
Smoking Status can be Never, Former, or Current.
Physical Activity can be Low, Medium, or High.
Diet Quality can be Poor, Average, or Good.
Target Variable: diabetes_diagnosis where 1 means Diabetic and 0 means Non-Diabetic.
Data Collection Methods: Electronic Health Records from hospitals, government health databases, annual health checkup camps, wearable devices like glucose monitors and smartwatches, and patient questionnaires.
Data Quality Risks:
Missing values can occur when patients skip certain tests. This can be handled by imputing with median values.
Class imbalance happens because there are more non-diabetic patients than diabetic ones. This can be fixed using class weights or oversampling with SMOTE.
Outdated records may not reflect current health. Only records from the last 5 years should be used.
Privacy concerns exist because patient data is sensitive. All records must be anonymized and must follow HIPAA and DPDP regulations.
Measurement errors can come from different labs using different equipment. Units and reference ranges should be standardized.
Task 5: Model Recommendation
Recommended Model: Feed-Forward Neural Network
Architecture:
Input Layer takes 11 features which are the patient health metrics.
Hidden Layer 1 has 64 neurons with ReLU activation, BatchNormalization, and Dropout of 0.3.
Hidden Layer 2 has 32 neurons with ReLU activation, BatchNormalization, and Dropout of 0.2.
Output Layer has 1 neuron with Sigmoid activation that gives the probability of diabetes.
Why This Model:
Feed-forward neural networks work excellently with tabular and structured data. Neural networks capture complex relationships between features such as BMI combined with age and glucose together. Sigmoid activation gives a risk score between 0 and 1, not just a yes or no answer. The model can handle millions of patient records and is scalable. SHAP values can be used to explain which features drove the prediction, making it interpretable for doctors.
Task 6: Evaluation Plan
Technical Metrics:
ROC-AUC should be greater than 0.85. This measures the ability to separate diabetic from non-diabetic patients.
Recall or Sensitivity should be greater than 0.90. The model must not miss actual diabetic patients because false negatives are dangerous in healthcare.
Precision should be greater than 0.80 to avoid too many false alarms.
F1-Score should be greater than 0.85 as a balance of precision and recall.
Accuracy should be greater than 0.85 for overall correctness.
Note: Recall is prioritized over precision in healthcare. It is worse to miss a sick patient than to send a healthy patient for extra tests.
Business Metrics:
Early detection rate measures the percentage of diabetes cases caught 6 or more months before traditional diagnosis.
Cost savings per patient measures the reduction in treatment cost due to early intervention.
Doctor time saved measures hours per week saved on manual screening.
Patient satisfaction measures feedback scores from patients and doctors.
Reduction in complications measures the percentage decrease in diabetic complications like blindness and kidney failure.
Possible Failure Cases:
A model trained on urban data may perform poorly on rural patients.
Rare patient profiles not seen in training data may be misclassified.
Data drift can happen when patient demographics change over time.
Sensor errors from wearable devices may cause wrong inputs.
Human Review Process:
All high-risk predictions with a score above 0.7 are flagged for mandatory doctor review. The model never makes a final diagnosis and is always used as a decision support tool. The clinical team conducts a monthly audit to review model predictions versus actual outcomes. Doctor corrections are fed back into the model for monthly retraining.
Task 7: Responsible AI Considerations
Bias in Data:
Training data may be biased toward certain demographics such as urban populations, males, or specific ethnicities. To mitigate this, diverse data should be collected and model performance should be audited separately for each demographic group.
Incorrect Predictions:
False negatives which means missing a diabetic patient can be life-threatening. False positives cause unnecessary anxiety and extra medical costs. To mitigate this, a high recall threshold should be set and mandatory doctor review should be required for borderline cases.
Privacy Concerns:
Patient health data is extremely sensitive and protected by law. To mitigate this, all data should be anonymized, federated learning should be used so that data never leaves the hospital server, and the system must comply with India's DPDP Act and international HIPAA standards.
Over-Reliance on AI:
Doctors may blindly trust AI predictions without applying clinical judgment. To mitigate this, AI should be framed as a second opinion tool, doctors should be trained on AI limitations, and confidence scores should always be shown rather than just predictions.
Impact on Patients:
Patients may be distressed by an AI-generated high risk label. To mitigate this, results should only be communicated by qualified doctors and a clear explanation of what the score means should always be provided.
Need for Human Oversight:
The AI model must be regularly audited and retrained. A dedicated AI governance committee should monitor the system. A clear escalation process must exist for when model confidence is low.
Task 8: Final Solution Summary
Problem: Diabetes is detected too late, causing severe complications. Manual screening is slow, expensive, and reactive.
Proposed AI Solution: A Feed-Forward Neural Network that analyzes 11 patient health features and predicts a diabetes risk score from 0 to 100 percent in real-time at the point of care.
Required Data: Structured patient records including age, BMI, glucose, HbA1c, blood pressure, insulin, family history, and lifestyle data. A minimum of 50,000 labeled patient records is needed for training.
Model Recommendation: Feed-Forward Neural Network with a 64 to 32 to 1 architecture with Sigmoid output and class weights for imbalance handling. Gradient Boosting using XGBoost is recommended as the baseline model.
Expected Business Impact: 40 percent reduction in late-stage diabetes diagnoses. 30 percent reduction in treatment costs. Screening time reduced from 3 days to under 1 minute.
Risks: Data bias, privacy breaches, over-reliance by doctors, and false negatives.
Mitigation Plan: Diverse training data, HIPAA and DPDP compliance, mandatory doctor review for high-risk cases, monthly model audits, and confidence scores shown to doctors at all times.
This solution is designed as a decision support tool. AI assists doctors but never replaces them.

STEP 8 - Now for the diagram. Click "Add file" then "Create new file". Type the name as: diagrams/solution_architecture.png
Wait, you cannot upload a PNG this way. Instead do this:
Click "Add file" then "Upload files". Then upload the diagram image file that you downloaded earlier. Make sure to first create the diagrams folder by naming a file as diagrams/placeholder.txt and writing anything inside it, then commit. After that go inside the diagrams folder and upload the PNG there.


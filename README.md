# Predicting Loan Creditworthiness
## Problem Description
Financial institutions must carefully balance risk and opportunity when approving loans.  The goal of this project is to predict loan creditworthiness—determining whether a loan should be approved—using machine learning models. By accurately identifying borrowers likely to repay, financial institutions can minimize defaults and reduce risk while extending credit to creditworthy customers.  

This solution:

- Analyzes raw loan application data from a Kaggle competition.

- Performs extensive exploratory data analysis (EDA) to understand the data distribution, missing values, duplicates, and relationships between features and the target.

- Implements robust feature engineering to derive meaningful features.

- Trains and tunes multiple models (including linear and tree-based methods) with calibration.

- Uses an automated pipeline to select the best model based on evaluation metrics and industry-standard thresholds.

- Deploys the model via a Flask API, containerized with Docker, and includes Kubernetes manifests for cloud deployment.
---
## Data Description  
The dataset is downloaded from [Kaggle's Bluechip Summit Credit Worthiness Prediction Competition](https://www.kaggle.com/competitions/bluechip-summit-credit-worthiness-prediction/data). The raw data includes:
-  **ID:** Unique identifier for each observation.
-  **Loan_ID:** A unique loan ID.
-  **Gender:** Either male or female.
-  **Married:** Whether the applicant is married (yes/no).
-  **Dependents:** Number of persons depending on the client.
-  **Education:** Applicant education (Graduate or Undergraduate).
-  **Self_Employed:** Self-employed status (Yes/No).
-  **ApplicantIncome:** Applicant income.
-  **CoapplicantIncome:** Co-applicant income.
-  **LoanAmount:** Loan amount in thousands.
-  **Loan_Amount_Term:** Term of the loan in months.
-  **Credit_History:** Whether the applicant's credit history meets guidelines (1 = yes, 0 = no).
-  **Property_Area:** The area where the applicant lives (Urban, Semi-Urban, or Rural).
-  **Loan_Status:** Target variable; loan approved (Y/N).
-  **Total_Income:** Total income, presumably sum of incomes from applicant and coapplicant.

The README and associated documentation include details on each field and how they are processed.

---
## Exploratory Data Analysis (EDA)

The `EDA.py` script performs an extensive exploratory analysis including:

  

-  **Missing Values and Duplicates:** Checking for missing data and duplicate entries.
-  **Statistical Summary:** Summary statistics (min, max, mean, std) for numerical columns.
-  **Distribution Analysis:** Histograms and box plots for numerical features, and count plots for categorical features.
-  **Correlation Analysis:** A correlation heatmap to visualize relationships among numeric features.
-  **Target Analysis:** Distribution of the target variable (`Loan_Status`) and its relation to other features via box plots and grouped count plots.

Run the EDA using:

```bash
pipenv  run  python  EDA.py
```

## Model  Training  and  Tuning

The  `train.py`  script  includes  the  following:|

> **Data  Preprocessing & Feature  Engineering**
> The  raw  data  is  read,  processed  with  feature_engineering.py,  and  split  into  features  and  target.

> **Cross-Validation  and  Calibration**
> Multiple  models (Logistic Regression,  Random  Forest,  Decision  Tree,  Gradient  Boosting,  and  XGBoost) are trained using stratified 5-fold cross-validation.

> For  each  fold,  raw  predicted  probabilities  are  collected  and  then  calibrated  using  a  `Spline  Calibration  method (implemented in  the  SplineCalibration  class)`.

> The  optimal  decision  threshold  for  each  model  is  computed  using  the  precision-recall  curve  and  maximizing  the  F1  score.

## Final  Model  Training  and  Evaluation 

The  best  models  are  retrained  on  the  full  training  data. Models  are  evaluated  on  a  hold-out  validation  set  with  metrics  like  accuracy,  precision,  recall,  and  F1  score through StratifiedKFold due to the target imbalance.

> **Artifact  Saving**
> The  best  model  artifacts (e.g., model,  DictVectorizer,  optimal  threshold) are saved using joblib for  deployment  in  the  prediction  service.

Run  the  training  script
```bash
pipenv  run  python  train.py
```
## Model  Deployment

> **Flask  API**
> The  `predict.py`  script  deploys  the  best  model  as  a  Flask  API.  It  loads  the  saved  artifacts (e.g., best_model.pkl,  best_dv.pkl,  and  optimal_threshold.pkl), applies feature engineering to incoming raw data, and then  makes  predictions.
> **Testing  the  API**
> After  running  the  prediction  service,  you  can  test  it  using  a  POST  request.  For  example,  using  Python’s  requests:

```python
import  requests
url  =  'http://localhost:9990/predict'
data  =  {

"Gender":  "Male",

"Married":  1,

"Dependents":  "0",

"Self_Employed":  0,

"Loan_Amount_Term":  360,

"Credit_History":  1,

"Property_Area":  "Urban",

"ApplicantIncome":  5000,

"CoapplicantIncome":  1500.0,

"LoanAmount":  150,

"Education":  1

}
response  =  requests.post(url,  json=data).json()
print(response)
```

Run  the  prediction  API:

```bash
pipenv  run  python  predict.py
```

## Containerization  with  Docker

A  `Dockerfile`  is  provided  to  containerize  the  application. To  build  and  run  the  Docker  container:
> **Build  the  Docker  image**
> ```bash
> docker  build  -t  loan_credit_predict  .
> ```
> **Run  the  Docker  container**
> ```bash
> docker  run  --rm  -p  9990:9990  loan_credit_predict
> ```

## Cloud  Deployment (Kubernetes)

For  cloud  deployment,  Kubernetes  manifests (deployment.yaml and  service.yaml) are provided in the repository. The following steps were utilized:
>**Deploy  to  Kubernetes  cluster**
>```bash
>kubectl  apply  -f  deployment.yaml
>kubectl  apply  -f  service.yaml
>```
>**Obtain external IP**
> ```bash
> kubectl  get  service  loan-credit-predict-service
> ```
> **Test external IP**
> ```bash
> pipenv run python predict_loan_worthiness.py
> ```
> ![Test Running Cloud Deployment](https://tinyurl.com/5xpawyza)

## Dependency  and  Environment  Management

This  project  uses  Pipenv  for  dependency  management.  The  repository  includes  a  Pipfile  and  Pipfile.lock  to  ensure  reproducibility.  To  install  the  dependencies  and  activate  the  virtual  environment,  run:
```bash
pipenv  install
```
```bash
pipenv  shell
````

## Containerization

The  provided  Dockerfile  builds  a  Docker  image  using  pipenv  to  install  all  dependencies.  This  image  can  be  used  to  run  the  training  or  prediction  services  in  any  environment  that  supports  Docker.

## Cloud  Deployment

Cloud  deployment  is  facilitated  via  Kubernetes.  The  included  `deployment.yaml` and  `service.yaml`  files  provide  a  template  for  deploying  the  containerized  Flask  API  to  a  cloud  provider  or  a  local  Kubernetes  cluster (such as  Minikube  or  Docker  Desktop’s  Kubernetes). Detailed instructions for deployment are included in this README and optionally in a separate Deployment.md. 

Exported  Scripts
The  project  is  modularized  as  follows:
`EDA.py`  –  Contains  all  functions  for  exploratory  data  analysis.
`feature_engineering.py`  –  Implements  feature  engineering  transformations.
`train.py`  –  Handles  model  training,  calibration,  evaluation,  and  saving  artifacts.
`predict.py`  –  Implements  a  Flask  API  for  serving  predictions.
`Dockerfile`  –  Containerizes  the  application.
`Kubernetes  Manifests` -  `deployment.yaml`  and  `service.yaml`  for  cloud  deployment.

## Reproducibility

> **Data  Access**
> The  dataset  is  downloaded  from  Kaggle and included in the repository.
> **Scripts**
> All  necessary  scripts  are  included.  The  training  pipeline  and  prediction  API  are  fully  modularized.
> **Environment**
> Pipenv  is  used  to  ensure  that  the  exact  dependency  versions  are  installed.
> **Instructions**
> Detailed  instructions  for  running  the  code  locally,  in  Docker,  and  in  Kubernetes  are  provided.

## Usage  Instructions

**Running  locally  with  Pipenv** 
>Activate  the  Environment:
>```bash
>pipenv  shell
>```
>**Run  the  Training  Script**
>```bash
> pipenv run python  train.py
> ```
> It performs  feature  engineering,  trains  multiple  models  with  calibration,  evaluates  them,  and  saves  the  best  model  artifacts (best_model.pkl, best_dv.pkl,  and  optimal_threshold.pkl).
> **Run  the  Prediction  Service**
> ```bash
> pipenv run python  predict.py
> ```
> The  Flask  API  will  start  on  port  9990 which can be tested  with  a  POST  request.

**Running  via  Docker** 
>**Build  the  Docker  Image**
>```bash
>docker  build  -t  loan_credit_predict  .
>```
>**Run  the  Docker  Container**
>```bash
>docker  run  --rm  -p  9990:9990  loan_credit_predict
>```

**Deploying  to  Kubernetes**
> **Build & Push  Docker  Image (If  using  a  public/private  registry)**
>  ```bash
>  docker  build  -t  your-dockerhub-username/loan_credit_predict:latest  .
>  docker  push  your-dockerhub-username/loan_credit_predict:latest
>  ```

>**Apply  Manifests**
>```bash
>kubectl  apply  -f  deployment.yaml
>kubectl  apply  -f  service.yaml
>```
>**Check  Service**
>```bash
>kubectl  get  service  loan-credit-predict-service
>```

Use  the obtained  external IP  or  NodePort  to  test the  API.

## License

This  project  is  licensed  under  the  MIT  License.

## Contact

For  questions  or  feedback,  please  contact [Austine]  at [austinnoah92@gmail.com].

---
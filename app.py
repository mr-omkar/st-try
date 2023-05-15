import pandas as pd
import numpy as np
import dill

df = pd.read_csv("training_set.csv")
df["Gender"]= df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"]= df["Married"].fillna(df["Married"].mode()[0])
df["Dependents"]= df["Dependents"].fillna(df["Dependents"].mode()[0])
df["Self_Employed"]= df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["Education"]= df["Education"].fillna(df["Education"].mode()[0])


df["ApplicantIncome"]= df["ApplicantIncome"].fillna(df["ApplicantIncome"].mean())
df["CoapplicantIncome"]= df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].mean())
df["LoanAmount"]= df["LoanAmount"].fillna(df["LoanAmount"].mean())
df["Loan_Amount_Term"]= df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"]= df["Credit_History"].fillna(df["Credit_History"].median())

df["Credit_History"] = df["Credit_History"].astype(object)
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype(object)

ar = df.head(1).drop(["Loan_ID","Gender","Loan_Status"],axis=1)
ar["LoanAmount"] = df["LoanAmount"].mean()
print(ar.values)


m1 = dill.load(open("loanPrepo_p1.pkl","rb"))
r = m1.transform(ar)
print(r)

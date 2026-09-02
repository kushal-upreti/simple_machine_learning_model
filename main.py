from ml_model_class import Linear_Regression_Ml
import pandas as pd

student_df = pd.read_csv("/data/Files/linreg_students.csv")

length = len(student_df)
train_length = int(0.8* length)
x_train = []
y_train = []
x_test = []
y_test = []


for i in range(0, train_length):
    x_train.append(student_df['hours_studied'].iloc[i])
    y_train.append(student_df['exam_score'].iloc[i])
    
for i in range(train_length, length):
    x_test.append(student_df['hours_studied'].iloc[i])
    y_test.append(student_df['exam_score'].iloc[i])


obj1 = Linear_Regression_Ml()
obj1.fit(x_train, y_train)
slope = obj1.slope
intercept = obj1.intercept
obj1.predict(7.5)
predicted_score = obj1.predicted_value
obj1.ml_mse_r2_score(x_test, y_test)
mse = obj1.avg_squared_error
r2_score = obj1.r2_value

print(f"\n-----------linear regression ml model result--------------- \nSlope: {slope} \n Intercept: {intercept} \nPredicted Score: {predicted_score}\n Mean Square Error: {mse}\n R2 score: {r2_score}")
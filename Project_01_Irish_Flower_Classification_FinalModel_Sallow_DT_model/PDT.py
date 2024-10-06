import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV


from sklearn import metrics
import sklearn.metrics as skmet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

from sqlalchemy import create_engine
import joblib
import pickle

df = pd.read_excel('C:/Users/SHINU RATHOD/Desktop/internship assignment/CipherByte Technologies/01_Irish_Flower_Classification/Dataset/Iris Flower.xlsx')
df.head()

engine = create_engine('mysql+pymysql://root:1122@localhost/cipherbyte_internship')
df.to_sql('irish_flower_classification', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
sql = 'select * from irish_flower_classification;'
df = pd.read_sql_query(sql, engine) 
df.head()
df.info()
df.describe()

# df.isnull().sum().sum()   # ther is no missing or null values present
# df.duplicated().sum()     # there is no duplicate record  present in dataset



# # ############################# univariate analysis
# # # 1. univariate analysis
# df.columns
# df['Species'].unique()
# df['Species'].value_counts()     # since all unique instances have equal records so dataset is balance dataset
# sns.countplot(df["Species"])


# # # Histogram # Visualize the distribution and frequency of the target variable
# plt.hist(df['Species'], bins=5, color='skyblue', edgecolor='red')
# plt.title('Histogram of Species')
# plt.xlabel('Species')
# plt.ylabel('Frequency')
# plt.show()


# # # Boxplot
# df.boxplot(column=['SepalLengthCm'])    # here we can see that there is no outliers presint in dataset
# plt.title('Boxplot')
# plt.show()

# df.columns


# # # Q-Q Plot (Quantile-Quantile Plot):    
# from scipy import stats
# stats.probplot(df['SepalLengthCm'], dist="norm", plot=plt)    #dataset is completly normally distributed
# plt.show()


# # # Shapiro-Wilk Test:
# from scipy.stats import shapiro
# stat, p = shapiro(df['SepalLengthCm'])
# print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
# # test statistic value, which typically ranges between 0 and 1. A value closer to 1 indicates that the data is more likely to be normally distributed.
# # The p-value is used to determine the significance of the test. It helps in deciding whether to reject the null hypothesis
# # . p-value is less than 0.05, you reject the null hypothesis (H₀). This means that the data is not normally distributed.

# # # 3. Skewness and Kurtosis:
# df['SepalLengthCm'].skew()
# df['SepalLengthCm'].kurt()
# # skewness : -0.5 to 0 or 0 to 0.5: An approximately symmetrical distribution 
# # kurtosis : -1.4750228130423522 its mean its laptokurtosis

# nf = df.select_dtypes(exclude = 'object').columns
# cf = df.select_dtypes(include = 'object').columns



# # ######################### bivariate analysi
# corrmat = df[nf].corr()

# # Compute the correlation matrix
# sns.heatmap(corrmat, annot = True, cmap = "YlGnBu")  #camp = 'coolwarm'


# ### playing with AutoEDA Lib to check data quality
#  # # 1) SweetViz
# import sweetviz as sv
# s = sv.analyze(df)
# s.show_html()


# # # 2) D-Tale
# import dtale 
# d = dtale.show(df)
# d.open_browser()

### Extracting dependent and independent variables
x = df.drop(columns = ['Id', 'Species'])
y = df['Species']


x.isnull().sum().sum()   # 0 total missing or null values present in dataset
############################
# Define pipeline for missing data if any
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'median'))])
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, x.columns)])
x_imputed = preprocessor.fit(x)
joblib.dump(preprocessor, 'meanimpute')

x_imputed_df = pd.DataFrame(x_imputed.transform(x), columns = x.columns)
x_imputed_df
x_imputed_df.isnull().sum()



###################### playing with outliers
# Defining a function to count outliers present in dataset
def count_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    return outliers
# Counting outliers before applying Winsorization tech
outliers_before = x_imputed_df.apply(count_outliers)
outliers_before      
outliers_before.sum()  # here 4 total num of outlier/extreame values are present in dataset after imputing missing val with mean val

# plotting boxplot for to check outliers
x_imputed_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (35,20)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()

############################## Define Winsorization pipeline
# Define the model with percentiles:# Default values # Right tail: 95th percentile # Left tail: 5th percentile
winsorizer_pipeline = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
X_winsorized = winsorizer_pipeline.fit(x_imputed_df)
joblib.dump(winsorizer_pipeline, 'winsor')  

# Transform Winsorized data back to DataFrame
X_winsorized_df = pd.DataFrame(X_winsorized.transform(x_imputed_df), columns=x_imputed_df.columns)

# Count outliers after Winsorization
outliers_after = X_winsorized_df.apply(count_outliers)
outliers_after
#plotting boxplot after removing outliers
X_winsorized_df.plot(kind = 'box', subplots = True, sharey = False, figsize = (35, 25)) 
plt.subplots_adjust(wspace = 0.75)  
plt.show()



############################ creating pipline for standard scaler
scale_pipeline = Pipeline([('scale', StandardScaler())])
X_scaled = scale_pipeline.fit(X_winsorized_df)
joblib.dump(scale_pipeline, 'scale')

X_scaled_df = pd.DataFrame(X_scaled.transform(X_winsorized_df), columns = X_winsorized_df.columns)
X_scaled_df
clean_data = X_scaled_df


# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(clean_data, y, test_size=0.2, random_state=42)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape 

######################################### Decision Tree Model
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(criterion = 'entropy')
model.fit(xtrain, ytrain)
pickle.dump(model, open('dtbest.pkl', 'wb'))
# Prediction on Train Data
ytrain_preds = model.predict(xtrain)
 # Accuracy
print(accuracy_score(ytrain, ytrain_preds))
pd.crosstab(ytrain, ytrain_preds, rownames = ['Actual'], colnames = ['Predictions']) 
confusion_matrix(ytrain, ytrain_preds)
cr = classification_report(ytrain, ytrain_preds)


# Prediction on Test Data
ytest_preds = model.predict(xtest)
 # Accuracy
print(accuracy_score(ytest, ytest_preds))
pd.crosstab(ytest, ytest_preds, rownames = ['Actual'], colnames = ['Predictions']) 
confusion_matrix(ytest, ytest_preds)
classification_report(ytest, ytest_preds)



####################### model testing on new data
model1 = pickle.load(open('dtbest.pkl', 'rb'))
impute = joblib.load('meanimpute')
winzor = joblib.load('winsor')
minmax = joblib.load('scale')
# encode = joblib.load('encoding')
data = pd.read_excel('C:/Users/SHINU RATHOD/Desktop/internship assignment/CipherByte Technologies/01_Irish_Flower_Classification/Dataset/Irish_Flower_test.xlsx')
 
x = df.drop(columns = ['Id', 'Species'])
y = df['Species']


x_impute = pd.DataFrame(impute.transform(data), columns = x.columns)
x_winz = pd.DataFrame(winzor.transform(x_impute), columns = x_impute.columns)
x_scale = pd.DataFrame(minmax.transform(x_winz), columns = x_winz.columns)

# x_encode = pd.DataFrame(encode.transform(data), columns = encode.get_feature_names_out())
# clean = pd.concat([x_scale, x_encode], axis=1)
prediction = pd.DataFrame(model1.predict(x_scale), columns = ['Pred'])
prediction
final = pd.concat([prediction, data], axis = 1)
final['Pred'].value_counts()

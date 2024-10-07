import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

import calendar

from sqlalchemy import create_engine

UE_in_india = pd.read_excel('C:/Users/SHINU RATHOD/Desktop/internship assignment/07_CipherByte Technologies/02_unemployement analysis/dataset/Unemployment in India.xlsx')
df = pd.read_excel('C:/Users/SHINU RATHOD/Desktop/internship assignment/07_CipherByte Technologies/02_unemployement analysis/dataset/Unemployment_Rate_upto_11_2020.xlsx')

engine = create_engine('mysql+pymysql://root:1122@localhost/cipherbyte_internship')
# df.to_sql('Unemployment_Rate_upto_11_2020', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
df.to_sql('irish_flower_classification', con=engine, if_exists='replace', chunksize=1000, index=False)

sql = 'select * from Unemployment_Rate_upto_11_2020;'
df = pd.read_sql_query(sql, engine) 

df.head()
df.sample(10)
df.tail()
df.dtypes
df.shape

df.info()   #754 entries or records and total we have 7 columns
df.describe()
df.columns


df.isnull().sum()
df.isnull().sum().sum()  # since dataset have 0 NA or missing values
df.duplicated().sum()  # we have 0 duplicated records in dataset

#################### data preprocessing
# seems like column name column names are not correct so we should have to rename the columns name 
df.columns= ["States", "Date", "Frequency", "Estimated Unemployment Rate", "Estimated Employed", "Estimated Labour Participation Rate","Region","longitude","latitude"]

#converting the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)

#converting the 'Frequency' column to categorical data type
df['Frequency']= df['Frequency'].astype('category')

#extracting the 'Month' from the Date
df['Month'] =  df['Date'].dt.month

#creating a new column 'MonthNumber' by converting the 'Month' column values to integers
df['MonthNumber'] = df['Month'].apply(lambda x : int(x))

#creating a new column 'MonthName' by converting the 'MonthNumber' column values to the monthNames
df['MonthName'] =  df['MonthNumber'].apply(lambda x: calendar.month_abbr[x])

#ensuring the categorical vairable
df['Region'] = df['Region'].astype('category')


nf = df.select_dtypes(include=['number']).columns
######EDA/DESCRIPTIVE STATISTICS
df.describe()

# 5-number summary of the numerical variables which give some information
summery = round(df[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']].describe().T,2)
#rounded to 2 decimal points and transposed to get a horizontal version

#grouping by 'Region' and finding mean values for the numerical columns
regionStats = df.groupby(['Region'])[['Estimated Unemployment Rate',
                                      'Estimated Employed',
                                      'Estimated Labour Participation Rate']].mean().reset_index()

#rounding the values to 2 decimal points
round(regionStats,2)



## correlation between the features of this dataset
# Use the correct style
print(plt.style.available)
plt.style.use('Solarize_Light2')

# Create the heatmap for the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df[nf].corr(), annot=True, cmap='coolwarm')  # `annot=True` adds the correlation values on the heatmap
plt.show()

###################### Graphical represention/visualization
############################# 1. univariate analysis
# boxplot
# playing with outliers ### Handling outliers that are present in the dataset
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
outliers_before = df[nf].apply(count_outliers)
outliers_before      
outliers_before.sum() 
 
# plotting boxplot for to check outliers
df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,10)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()


# histplot on variable Estimated Unemployment Rate
sns.histplot(df['Estimated Unemployment Rate'], kde=True)
plt.title('Distribution of Estimated Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.show()

# Countplot for Region
plt.figure(figsize=(6, 4))
sns.countplot(x='Region', data=df)
plt.title('Distribution of Areas')
plt.show()


################################## bivariate analysis
# 1. Correlation matrix for numerical columns
df.info()
correlation_matrix = df[nf].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# 2. Scatter plot for Estimated Unemployment Rate vs Estimated Employed
df.columns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Estimated Unemployment Rate'], y=df['Estimated Employed'])
plt.title('Estimated Unemployment Rate vs Estimated Employed')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Number of Employed')
plt.show()
# here we can see there is no correlation between this two variable


# 3. Line Plot (Time Series)
# Line plot of Date vs Estimated Unemployment Rate
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # Ensure Date is in datetime format

plt.figure(figsize=(10, 6))
sns.lineplot(x=df['Date'], y=df['Estimated Unemployment Rate'])
plt.title('Time Series of Estimated Unemployment Rate')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()
# visualize how "Estimated Unemployment Rate" changes over time.


# 4. Bar plot of Region vs Estimated Employed
# To analyze how "Area" affects variables like "Estimated Employed", a bar plot works well.
plt.figure(figsize=(8, 6))
sns.barplot(x=df['Region'], y=df['Estimated Employed'])
plt.title('Area vs Estimated Employed')
plt.xlabel('Area')
plt.ylabel('Estimated Employed')
plt.show()


# 5. Jointplot (Continuous vs. Continuous)
# Jointplot of Estimated Unemployment Rate vs Estimated Labour Participation Rate
sns.jointplot(x='Estimated Unemployment Rate', y='Estimated Labour Participation Rate', data=df, kind='reg')
plt.show()



######################### multi-variate analysis
# 1. Pairplot
# Pairplot for numerical variables
sns.pairplot(df[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']])
plt.show()
# pairplot to visualize pairwise relationships between multiple variables simultaneously.


# 2. 3D Scatter Plot (Multivariate Relationships)
# visualize the relationship between three variables using a 3D scatter plot.
from mpl_toolkits.mplot3d import Axes3D
# 3D Scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Estimated Employed'], df['Estimated Labour Participation Rate'], df['Estimated Unemployment Rate'], c='b', marker='o')

ax.set_xlabel('Estimated Employed')
ax.set_ylabel('Labour Participation Rate (%)')
ax.set_zlabel('Unemployment Rate (%)')

plt.title('3D Scatter Plot of Employment and Labour Participation vs Unemployment Rate')
plt.show()



# 3. Multivariate KDE Plot
# Corrected KDE plot code
sns.kdeplot(x=df['Estimated Unemployment Rate'], y=df['Estimated Employed'], cmap='Blues', shade=True, bw_adjust=.5)
plt.title('KDE Plot of Estimated Unemployment Rate vs Estimated Employed')
plt.show()



# Time series plot for Unemployment Rate and Labour Participation Rate over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Estimated Unemployment Rate'], label='Unemployment Rate', color='blue')
plt.plot(df['Date'], df['Estimated Labour Participation Rate'], label='Labour Participation Rate', color='green')
plt.legend()
plt.xticks(rotation=45)
plt.title('Unemployment Rate and Labour Participation Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Rate (%)')
plt.show()





# estimated number of employees according to different regions of India
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region", data=df)
plt.show()


# unemployment rate according to different regions of India
plt.figure(figsize=(12, 10))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=df)
plt.show()

# create a dashboard to analyze the unemployment rate of each Indian state by region. using a sunburst plot
unemploment = df[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()




# Auto EDA
# ---------
# 1. Sweetviz
# 2. Autoviz
# 3. Dtale
# 4. Pandas Profiling
# 5. Dataprep


######################### playing with AutoEDA
# # 1) SweetViz
import sweetviz as sv
s = sv.analyze(df)
s.show_html()

# # 2) D-Tale
import dtale 
d = dtale.show(df)
d.open_browser()




# Clean, Analyze, Visualize And Perform EDA With Klib Library
import klib
# df = pd.DataFrame(data)

df = df.drop(columns = ['Date'], axis=1)
# klib.describe - functions for visualizing datasets
klib.cat_plot(df) # returns a visualization of the number and frequency of categorical features
klib.corr_mat(df) # returns a color-encoded correlation matrix
klib.corr_plot(df) # returns a color-encoded heatmap, ideal for correlations
klib.corr_interactive_plot(df, split="neg").show() # returns an interactive correlation plot using plotly
klib.dist_plot(df) # returns a distribution plot for every numeric feature
klib.missingval_plot(df) # returns a figure containing information about missing values

# klib.clean - functions for cleaning datasets
klib.data_cleaning(df) # performs datacleaning (drop duplicates & empty rows/cols, adjust dtypes,...)
klib.clean_column_names(df) # cleans and standardizes column names, also called inside data_cleaning()
klib.convert_datatypes(df) # converts existing to more efficient dtypes, also called inside data_cleaning()
klib.drop_missing(df) # drops missing values, also called in data_cleaning()
klib.mv_col_handling(df) # drops features with high ratio of missing vals based on informational content
klib.pool_duplicate_subsets(df) # pools subset of cols based on duplicates with min. loss of information


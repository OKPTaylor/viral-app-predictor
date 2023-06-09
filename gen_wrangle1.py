import scipy.stats as stats
import pandas as pd
import os
import numpy as np

# Data viz:
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn stuff:
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression




'''---------------------------------------------------------------------- ACQUIRE ----------------------------------------------------------------------------'''


#This function will create a csv file 
def get_csv(csv_name):
    df = pd.read_csv(csv_name)
    print(f"CSV found")
    return df
# call should be: wrg.get_csv(csv_name)
'''---------------------------------------------------------------------- PREP --------------------------------------------------------------------------------'''
#fix everything in one go
def prep_it(df_name):
    df_name = clean_col(df_name)
    df_name.drop(columns=['app_id', 'currency', 'scraped_time',"minimum_installs", "maximum_installs", "minimum_android","last_updated","rating_count"], inplace=True)
    df_name.dropna(subset=['released',"installs","app_name"], inplace=True)
    #fill NaNs with right stuff
    values = {"developer_website":"no website", "privacy_policy":"no policy", "developer_id":"no id", "developer_email":"no email", "rating": 0}
    df_name.fillna(value=values, inplace=True)
    #in installs replace "+" with "" and change dtype to int
    df_name.installs = df_name.installs.str.replace("+", "").str.replace(",", "").astype(int)
    df_name["file_size"] = df_name["size"]
    df_name.drop(columns=["size"], inplace=True)
    df_name.file_size = df_name.file_size.str.replace("M", "000").str.replace("k", "").str.replace("Varies with device", "22500").str.replace(",","").str.replace("G","000000").astype(float)
    df_name.released = pd.to_datetime(df_name.released, format='%b %d, %Y')
    #drop rows with release year before 2020 and after 2021
    df_name = df_name[(df_name.released.dt.year >= 2020) & (df_name.released.dt.year < 2021)]
    col_list = ["content_rating","category"]
    df_name = encode(df_name, col_list)
    #replace developer_webiste values with boolean
    df_name.developer_website = df_name.developer_website.str.replace("no website", "False").str.replace("https://", "True").astype(bool)
    #replace privacy_policy values with boolean
    df_name.privacy_policy = df_name.privacy_policy.str.replace("no policy", "False").str.replace("http://", "True").astype(bool)
    #replace developer_email values with boolean
    df_name.developer_email = df_name.developer_email.str.replace("no email", "False").str.replace("mailto:", "True").astype(bool)
    #make new column for days since release
    df_name["days_since_release"] = (pd.to_datetime("2021-06-15") - df_name.released).dt.days
    #maken new column for rows with same developer_id thats boolean
    df_name["same_dev_id"] = df_name.duplicated(subset=["developer_id"])
    df_name.drop(columns=["developer_id"], inplace=True)
    #create new column for rows with a 1000000 or more dowloads thats boolean
    df_name["viral"] = df_name.installs >= 1000000
    #drop rows with less than 1000 downloads
    df_name = df_name[df_name.installs >= 1000]
    df_name = df_name.drop(columns=["installs","released","developer_email","developer_website","privacy_policy"])
    df_name = bool_to_int(df_name)
    return df_name

#fix ugly column names
def clean_col(df):
    new_col_name = []

    for col in df.columns:
        new_col_name.append(col.lower().replace('.', '_').replace(' ', '_'))

    df.columns = new_col_name

    df.head()
    return df
#call should be: wg.clean_col(df_name)

# encode the categorical column from a list of columns
def encode(df, col_list):
    for col in col_list:
        df = pd.concat([df, pd.get_dummies(df[col], drop_first=True)], axis=1)
        df.drop(columns=[col], inplace=True)
        
    return df

#call should be: wrg.encode(df_name, col_list)
#col_list should be a list of the columns you want to encode

#looks for nulls and returns columns with nulls and columns
def is_it_null(df_name):
    for col in df_name.columns:
        print(f"{df_name[col].isna().value_counts()}") 
#call should be: wrg.is_it_null(df_name)  


#removes all columns with more than 5% nulls, NaNs, Na, Nones. This should only be used when it makes sense to drop the column; may need to drop rows
def null_remove(df_name):
    for col in df_name.columns:
        if df_name[col].isna().value_counts("False")[0] < 0.95: #tests if a row cotains more than 5% nulls, NaNs, ect. 
            df_name.drop(columns=[col], inplace=True)
            print(f"Column {col} has been dropped because it contains more than 5% nulls")   
#call should be: wrg.null_remove(df_name)           

#brings back all the columns that may be duplicates of other columns
def col_dup(df_name):
    for col1 in df_name.columns:
        for col in df_name.columns:
            temp_crosstab=pd.crosstab(df_name[col] , df_name[col1])
            if temp_crosstab.iloc[0,0] != 0 and temp_crosstab.iloc[0,1] == 0 and temp_crosstab.iloc[1,0] == 0 and temp_crosstab.iloc[1,1] !=0:
                if col1 != col:
                    print(f"\n{col1} and {col} may be the duplicates\n")
                    
                    print(temp_crosstab.iloc[0:3,0:3])
                    print("--------------------------------------------")
       
#call should look like: wrg.col_dup(df_name)  

#changes bools to 1s and 0s
def bool_to_int(df_name):
    for col in df_name.columns:
        if df_name[col].dtype == 'bool':
            df_name[col] = df_name[col].astype(int)
    return df_name        

#the two following functions is for ploting univar
#makes a list of all var

def df_column_name(df_name):
    col_name = []
    for x in df_name.columns[2:]: #check to make sure the range is what you want
        col_name.append(x)   
    return col_name       

def plot_uni_var(df_name):    #plots univar
    for col in (df_column_name(df_name)):
        plt.hist(df_name[col])
        plt.title(col)
        plt.show()         

          
#splits your data into train, validate, and test sets for cat target var
def split_function_cat_target(df_name, target_varible_column_name):
    train, test = train_test_split(df_name,
                                   random_state=123, #can be whatever you want
                                   test_size=.20,
                                   stratify= df_name[target_varible_column_name])
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25,
                                   stratify= train[target_varible_column_name])
    return train, validate, test
#call should look like: 
#train_df_name, validate_df_name, test_df_name = wrg.split_function_cat_target(df_name, 'target_varible_column_name')

#splits your data into train, validate, and test sets for cont target var
def split_function_cont_target(df_name):
    train, test = train_test_split(df_name,
                                   random_state=123, #can be whatever you want
                                   test_size=.20)
                                   
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25)
    return train, validate, test
#call should look like: 
#train_df_name, validate_df_name, test_df_name = wrg.split_function_cont_target(df_name)
'''---------------------------------------------------------------------- EXPLORE --------------------------------------------------------------------------------'''

#This makes two lists containing all the categorical and continuous variables
def cat_and_num_lists(df_train_name):
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numeric varibles

    for col in df_train_name.columns[1:]: #make sure to set this to the range you want
        
        if df_train_name[col].dtype == 'O':
            col_cat.append(col)
        else:
            if len(df_train_name[col].unique()) < 4: #making anything with less than 4 unique values a catergorical value
                col_cat.append(col)
            else:
                col_num.append(col)
    print(f"The categorical variables are: \n {col_cat} \n") 
    print(f"The continuous variables are: \n {col_num} \n")                
    return col_cat , col_num           
#the call for this should be: wrg.cat_and_num_lists(df_train_name)

#feature selection using kbest
def select_kbest(x_train_scaled, y_train):
    kbest = SelectKBest(f_regression, k=2) #makes the k best using f_regression model
    kbest.fit(x_train_scaled, y_train) #fits it to x_train_scaled(if it's scalled) and y_train 
    kbest_results = pd.DataFrame(dict(p_value=kbest.pvalues_, f_score=kbest.scores_), index=x_train_scaled.columns)

    #makes a data frame with the p_values and f_scores
    kbest_results.sort_values(by=['f_score'], ascending=False)
    
    return kbest_results
    

#plots all pairwise relationships along with the regression line for each col cat and col num pair
def plot_variable_target_pairs(df_train_name,target_var):

    #df_train_name = df_train_name.sample(100000, random_state=123) #this is for sampling the data frame. This may not be needed for your data set
    col_cat, col_num = cat_and_num_lists(df_train_name)
    
    for col in col_num:
        print(f"{col.upper()} and {target_var}")
        
        sns.lmplot(data=df_train_name, x=col, y=target_var,
          line_kws={'color':'red'})
        plt.show()

#This plots all categorical variables against the target variable
def plot_categorical_and_target_var(df_train_name, target): #this defaults to 4 unique values
    col_cat, col_num = cat_and_num_lists(df_train_name)
    for col in col_cat:
        sns.barplot(x=df_train_name[col], y=df_train_name[target])
        plt.title(f"{col.lower().replace('_',' ')} vs {target}")
        plt.show()
        
#plots pairwise relationship of one feature with the target variable and T-test
def single_pairwise(df_train_name, target_var, col):
    print(f"Null Hypothesis: There is no linear correlation between {col.upper()} and {target_var}")
    print(f"{col.upper()} and {target_var}")
    sns.lmplot(data=df_train_name, x=col, y=target_var,
            line_kws={'color':'red'})
    plt.show()
    #runs a t-test on feature and target variable
    col_target = df_train_name[df_train_name[target_var] == 1][col]
    theoretical_mean = df_train_name[col].mean()
    t, p = stats.ttest_1samp(col_target, theoretical_mean)
    print(f"t = {t}, p = {p}, alpha = .05")
    if p < 0.05:
        print("We reject the null hypothesis")
        print("--------------------------------------------------------------------")
    else:
        print("We fail to reject the null hypothesis")
        print("--------------------------------------------------------------------")

   

    
        
#plots pairplots 
def pairplot_everything(df_train_name, cat):
    #df_train_name = df_train_name.sample(10000, random_state=123) #this is for sampling the data frame
    sns.pairplot(data=df_train_name, corner=True, hue=cat)
     

def corr_heatmap(df_train_name):
    #df_train_name = df_train_name.sample(10000, random_state=123) #this is for sampling the data frame
    plt.figure(figsize=(12,10))
    sns.heatmap(df_train_name.corr(), cmap='Blues', annot=True, linewidth=0.5, mask= np.triu(df_train_name.corr())) 
    plt.show()   


#This function is for running through catagorical on catagorical features graphing and running the chi2 test on them 
def cat_on_cat_graph_loop(df_train_name, target_ver, target_ver_column_name):
    col_cat, col_num = cat_and_num_lists(df_train_name)
    list_of_rejected_features = []
    for col in col_cat:
        print()
        print(col.upper())
        print(df_train_name[col].value_counts())
        print(df_train_name[col].value_counts(normalize=True))
        df_train_name[col].value_counts().plot.bar()
        plt.show()
        print()
        print()
        print(f'HYPOTHESIZE')
        print(f"H_0: {col.lower().replace('_',' ')} does not affect {target_ver}")
        print(f"H_a: {col.lower().replace('_',' ')} affects {target_ver}")
        print()
        print(f'VISUALIZE')
        sns.barplot(x=df_train_name[col], y=df_train_name[target_ver_column_name])
        plt.title(f"{col.lower().replace('_',' ')} vs {target_ver}")
        plt.show()
        print()
        print('ANALYZE and SUMMARIZE')
        observed = pd.crosstab(df_train_name[col], df_train_name[target_ver_column_name])
        chi2Test(observed,col, list_of_rejected_features)
        print()
        print("--------------------------------------------------------------------------------------------------------------------------")
    return list_of_rejected_features  
#the call should be: gw.cat_on_cat_graph_loop(dataframe_train_name,"target_ver", "target_ver_column_name")        

#this funciton works in this module to run the chi2 test with the above function
def chi2Test(observed,col, list_of_rejected_features=[]):
    
    alpha = 0.05
    chi2, pval, degf, expected = stats.chi2_contingency(observed)
    print('Observed')
    print(observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p-value = {pval:.4f}')
    print('----')
    if pval < alpha:
        print ('We reject the null hypothesis.')
    else:
        print ("We fail to reject the null hypothesis.")
        #append to list of rejected features with lower case
        list_of_rejected_features.append(col) 
        return list_of_rejected_features
# prep.chi2Test(observed) is the call 

#a function that tests for normality of target variable using the Shapiro-Wilk test
def normality_test(df_train_name, target_var):
    alpha = 0.05
    stat, p = stats.shapiro(df_train_name[target_var])
    print(f'Statistics={stat}, p={p}' )
    if p > alpha:
        print('The data is normally distributed (fail to reject H0)')
    else:
        print('The sample is NOT normally distributed (reject H0)')

# a function that plots a categorical variable against the target variable and runs a chi2 test
def cat_on_target(df_train_name, target_var, target_var_column_name, col):
    
    print(f"Null Hypothesis: {col.upper()} does not affect {target_var}")
    print()
    print(f"{col.upper()} and {target_var}")
    
    sns.barplot(x=df_train_name[col], y=df_train_name[target_var_column_name])
    plt.title(f"{col.lower().replace('_',' ')} vs {target_var}")
    plt.show()
    print()
    print('ANALYZE and SUMMARIZE')
    observed = pd.crosstab(df_train_name[col], df_train_name[target_var_column_name])
    chi2Test(observed,col)
    print()
    print("--------------------------------------------------------------------------------------------------------------------------")

    


#This funcition runs through the continuous varaibles and the continuous target variable and runs the pearsonr test on them
def pearsonr_loop(df_train_name, target_var, cat_count=4):
    alpha = 0.05
    col_cat, col_num = cat_and_num_lists(df_train_name, cat_count)
    for col in col_num:
        sns.regplot(x=df_train_name[col], y=df_train_name[target_var], data=df_train_name, line_kws={"color": "red"})
        plt.title(f"{col.lower().replace('_',' ')} vs {target_var}")
        plt.show()
        print(f"{col.upper()} and {target_var}")
        corr, p = stats.pearsonr(df_train_name[col], df_train_name[target_var])
        print(f'corr = {corr}')
        print(f'p = {p}')
        if p < alpha:
            print('We reject the null hypothesis, there is a linear relationship between the variables\n')
        else:
            print('We fail to reject the null hypothesis, there is not a linear relationship between the variables\n') 
        


#This function runs through the continuous variables and the continuous target variable and runs the spearman test on them
def spearman_loop(df_train_name, target_var, cat_count=4):
    alpha = 0.05
    col_cat, col_num = cat_and_num_lists(df_train_name, cat_count)
    for col in col_num:
        sns.regplot(x=df_train_name[col], y=df_train_name[target_var], data=df_train_name, line_kws={"color": "red"})
        plt.title(f"{col.lower().replace('_',' ')} vs {target_var}")
        plt.show()
        print(f"{col.upper()} and {target_var}")
        corr, p = stats.spearmanr(df_train_name[col], df_train_name[target_var])
        print(f'corr = {corr}')
        print(f'p = {p}')
        if p < alpha:
            print('We reject the null hypothesis, there is a linear relationship between the variables\n')
        else:
            print('We fail to reject the null hypothesis, there is not a linear relationship between the variables\n')    


#This function takes the data and scales it using the MinMaxScaler
def scale_data(train, validate, test):
    to_scale=train.columns.tolist()

    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    #this scales stuff 
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled 
# call should be train_scaled, validate_scaled, test_scaled = wrg.scale_data(x_train, x_validate, x_test)



#function for spliting the data into X and y for App Predictor Project only
def x_y_split(df_train_name, df_validate_name, df_test_name):
    list_of_rejected_features=['Beauty', 'Board', 'Card', 'Casino', 'Comics', 'Communication', 'Dating', 'Events', 'House & Home', 'Libraries & Demo', 'Maps & Navigation', 'Medical', 'Music', 'Parenting', 'Sports', 'Trivia', 'Word']
    X_train = df_train_name.iloc[:, 1:-1]
    X_train = X_train.drop(columns=list_of_rejected_features)
    y_train = df_train_name.viral
    X_validate = df_validate_name.iloc[:, 1:-1]
    X_validate = X_validate.drop(columns=list_of_rejected_features)
    y_validate = df_validate_name.viral
    X_test = df_test_name.iloc[:, 1:-1]
    X_test = X_test.drop(columns=list_of_rejected_features)   
    y_test = df_test_name.viral
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#function for spliting the data into X and y for App Predictor Project only for top featurs
def x_y_split_top_10(train_df, validate_df, test_df):
    X_train = train_df[['in_app_purchases','file_size','editors_choice','Everyone 10+','Everyone','Weather','Simulation','rating','Role Playing','ad_supported']]
    y_train = train_df.viral
    X_validate = validate_df[['in_app_purchases','file_size','editors_choice','Everyone 10+','Everyone','Weather','Simulation','rating','Role Playing','ad_supported']]
    y_validate = validate_df.viral
    X_test = test_df[['in_app_purchases','file_size','editors_choice','Everyone 10+','Everyone','Weather','Simulation','rating','Role Playing','ad_supported']] 
    y_test = test_df.viral

    return X_train, y_train, X_validate, y_validate, X_test, y_test
    
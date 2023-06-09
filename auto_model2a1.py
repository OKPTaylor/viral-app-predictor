import pandas as pd
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

def print_cm_metrics(conf): #function for calculating and formating Metrics for models
    tn, fp, fn, tp = conf.ravel()

    accuracy = (tp + tn)/(tn + fp + fn + tp)

    true_positive_rate = tp/(tp + fn)
    false_positive_rate = fp/(fp + tn)
    true_negative_rate = tn/(tn + fp)
    false_negative_rate = fn/(fn + tp)

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2*(precision*recall)/(precision+recall)

    support_pos = tp + fn
    support_neg = fp + tn

    dict = {
        'metric' : ['accuracy'
                    ,'true_positive_rate'
                    ,'false_positive_rate'
                    ,'true_negative_rate'
                    ,'false_negative_rate'
                    ,'precision'
                    ,'recall'
                    ,'f1_score'
                    ,'support_pos'
                    ,'support_neg']
        ,'score' : [accuracy
                    ,true_positive_rate
                    ,false_positive_rate
                    ,true_negative_rate
                    ,false_negative_rate
                    ,precision
                    ,recall
                    ,f1_score
                    ,support_pos
                    ,support_neg]
    }
    return pd.DataFrame(dict)

"""Use to generate a dataframe of knn train and validate scores and difference"""
def auto_knn_metrics(k_range, X_train, y_train, X_validate, y_validate): #for default k_range should be set to 5
    metrics = []

    for k in k_range:
        #make it
        knnx = KNeighborsClassifier(n_neighbors=k) #increases the k number
        #fit it
        knnx.fit(X_train, y_train)
        
        train_acc = knnx.score(X_train, y_train)
        val_acc = knnx.score(X_validate, y_validate)
        
        output = {"k": k,
                "train_accuracy": train_acc,
                "validate_accuracy": val_acc}
        
        metrics.append(output)
        
        eval_df = pd.DataFrame(metrics)
        eval_df['difference'] = eval_df['train_accuracy'] - eval_df['validate_accuracy']

        print(eval_df)
#call is: atm.auto_knn_metrics(k_range, X_train, y_train, X_validate, y_validate)


''' generates both train and validate graphs and metrics  '''
def auto_knn_graph(X_train, y_train, X_validate, y_validate, k_range=[3,5]): # k_range default should be set to 5
    scores_all = []
    

    for x in k_range:
        
        #make it
        knnx = KNeighborsClassifier(n_neighbors=x) #increases the k number
        #fit it
        knnx.fit(X_train, y_train)
        #transform it
        train_acc = knnx.score(X_train, y_train)
        y_pred = knnx.predict(X_train)
        
        conf = confusion_matrix(y_train, y_pred)
        
        print(f"\n------------------------ Train Model with K range of {x} ------------------------------")
        #plot_confusion_matrix(knnx, X_train, y_train)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 
        
        #evaluate on my validate data
        val_acc = knnx.score(X_validate, y_validate)
        y_vpred = knnx.predict(X_validate)
        conf = confusion_matrix(y_validate, y_vpred)
        
        print(f"\n---------------------Validate Model with K range of {x}---------------------------------")
        #plot_confusion_matrix(knnx, X_validate, y_validate)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf))
        print(f'\nk number = {x} accuracy = {train_acc, val_acc}')

        scores_all.append([x, train_acc, val_acc])
        

#call should be: atm.auto_model_knn(X_train, y_train, X_validate, y_validate, k_range)        

def auto_tree(x_train, y_train, x_validate, y_validate, max_depth_range=[5,10,13,15,20]):
    scores_all = []

    for x in max_depth_range:

        tree = DecisionTreeClassifier(max_depth=x) #creates, fits, scores train data
        tree.fit(x_train, y_train)
        train_acc = tree.score(x_train, y_train)
        y_pred = tree.predict(x_train)

        #conf = confusion_matrix(y_train, y_pred)
        
       
        print(f"\n------------------------ Train Model with depth of {x} ------------------------------")
        #plot_confusion_matrix(tree, x_train, y_train)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 
        
        #evaluate on my validate data
        val_acc = tree.score(x_validate, y_validate)
        y_vpred = tree.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        print(f"\n---------------------Validate Model with depth of {x}---------------------------------")
        #plot_confusion_matrix(tree, x_validate, y_validate)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_vpred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf))
        print(f'\nk number = {x} accuracy = {train_acc, val_acc}')

        scores_all.append([x, train_acc, val_acc])

#call should be: atm.auto_tree(x_train, y_train, x_validate, y_validate, max_depth_range)  

def auto_random_trees(x_train, y_train, x_validate, y_validate, max_depth=range(5,20)):
    scores_all = []

    for x in max_depth:
        
        #make it
        rf = RandomForestClassifier(random_state=123, max_depth=x) #increases the sample leaf whild decreasing max depth
        #fit it
        rf.fit(x_train, y_train)
        #transform it
        train_acc = rf.score(x_train, y_train)
        y_pred = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
    
        print(f"\n------------------------ Train Model with depth of {x} ------------------------------")
        #plot_confusion_matrix(rf, x_train, y_train)
        #plt.show()
        print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 
        
        #evaluate on my validate data
        val_acc = rf.score(x_validate, y_validate)
        y_vpred = rf.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        print(f"\n---------------------Validate Model with depth of {x}---------------------------------")
        #plot_confusion_matrix(rf, x_validate, y_validate)
        #plt.show()
        print(pd.DataFrame(classification_report(y_validate, y_vpred, output_dict=True)))
        print("------------ Metrics ----------")
        #print(print_cm_metrics(conf))
        print(f'\ndepth = {x} accuracy = {train_acc, val_acc}')

        scores_all.append([x, train_acc, val_acc])

def auto_lo_regress(x_train, y_train, x_validate, y_validate, max_depth=range(5,20)):
        scores_all = []

        for x in max_depth:
        
        #make it
            logit = LogisticRegression() #increases the sample leaf whild decreasing max depth
                #fit it
            logit.fit(x_train, y_train)
                #transform it
            train_acc = logit.score(x_train, y_train)
            y_pred = logit.predict(x_train)
            #conf = confusion_matrix(y_train, y_pred)
            
            print(f"\n------------------------ Train Model with C of {x} ------------------------------")
            #plot_confusion_matrix(logit, x_train, y_train)
            #plt.show()
            print(pd.DataFrame(classification_report(y_train, y_pred, output_dict=True)))
            print("------------ Metrics ----------")
            #print(print_cm_metrics(conf)) 
                
                #evaluate on my validate data
            val_acc = logit.score(x_validate, y_validate)
            y_vpred = logit.predict(x_validate)
            #conf = confusion_matrix(y_validate, y_vpred)

            print(f"\n---------------------Validate Model with C of {x}---------------------------------")
            #plot_confusion_matrix(logit, x_validate, y_validate)
            #plt.show()
            print(pd.DataFrame(classification_report(y_validate, y_vpred, output_dict=True)))
            print("------------ Metrics ----------")
            #print(print_cm_metrics(conf))
            print(f'\nk number = {x} accuracy = {train_acc, val_acc}')

            scores_all.append([x, train_acc, val_acc])


#function to test the best max_depth for the random forest model
def auto_random_trees_test(x_test, y_test, x_train, y_train, max_depth):
    #make it
    rf = RandomForestClassifier(max_depth, random_state=123) #increases the sample leaf whild decreasing max depth
    #fit it
    rf.fit(x_train, y_train)
        #transform it
        #train_acc = rf.score(x_train, y_train)
    y_pred_train = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
        #transform it
    test_acc = rf.score(x_test, y_test)
        
    y_pred = rf.predict(x_test)
        #conf = confusion_matrix(y_test, y_pred)
    
    print(f"\n------------------------ Test Model with depth of {max_depth} Scores------------------------------")
        #plot_confusion_matrix(rf, x_test, y_test)
        #plt.show()
    print(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
    print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 

    print(f"Accuracy is {test_acc}")

    plt.figure(figsize=(16,8))

    plt.hist(y_train, color='blue', alpha=.5, label="Actual App Viral")
    plt.hist(y_pred_train, color='red', alpha=.5, label="Model: Random Forest")
       #plt.hist(y_validate.G3_pred_lars, color='purple', alpha=.5, label="Model: Lasso Lars")
        #plt.hist(y_validate.G3_pred_glm, color='yellow', alpha=.5, label="Model: TweedieRegressor")
        #plt.hist(y_validate.G3_pred_lm2, color='green', alpha=.5, label="Model 2nd degree Polynomial")

    plt.xlabel("Viral")
    plt.ylabel("Number of Apps")
    plt.title("Comparing the Distribution of Viral Apps to Distributions of Predicted Viral Apps for the Top Model")
    plt.legend()
    plt.show()
        
 # Prediction .csv creation

'''predictions = pd.concat([
        pd.DataFrame(customer_id).reset_index().drop(columns='index'),
        pd.DataFrame(test_pred).rename(columns={0:'binn prediction'}),
        pd.DataFrame(rf.predict_proba(
            x_test)*100).drop(columns=0).rename(
            columns={1:'Probability of proper binning'})
                    ],axis=1)
    predictions.to_csv("Binning_predictions.csv")'''

'''----------------------------------------------- Auto Modeling No Graphs ---------------------------------------------------------------------------------'''

#This function runs through a range of k values and returns the accuracy scores for each k value       
def auto_knn_scores(X_train, y_train, X_validate, y_validate, k_range=[3,4,5]): # k_range default should be set to 5
    scores_all = []
    

    for x in k_range:
        
        #make it
        knnx = KNeighborsClassifier(n_neighbors=x) #increases the k number
        #fit it
        knnx.fit(X_train, y_train)
        #transform it
        train_acc = knnx.score(X_train, y_train)
        y_pred = knnx.predict(X_train)
        
        conf = confusion_matrix(y_train, y_pred)
        
        #evaluate on my validate data
        val_acc = knnx.score(X_validate, y_validate)
        y_vpred = knnx.predict(X_validate)
        conf = confusion_matrix(y_validate, y_vpred)
        
        scores_all.append([x, train_acc, val_acc])
    #create a dataframe from the scores_all list of lists
    df_scores = pd.DataFrame(scores_all)
    #add column names to the dataframe
    df_scores.columns = ['k', 'train_accuracy', 'validate_accuracy']
    # sort the dataframe by the difference between train and validate
    df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
    print(df_scores.sort_values(by='delta', ascending=False))

#call should be: atm.auto_knn_no_graph(X_train, y_train, X_validate, y_validate, k_range) 

#This function runs through a range of max_depth values and returns the accuracy scores for each max_depth value
def auto_random_forest_scores(x_train, y_train, x_validate, y_validate, max_depth=range(5,20)):
    scores_all = []

    for x in max_depth:
        
        #make it
        rf = RandomForestClassifier(random_state=123, max_depth=x) #increases the sample leaf whild decreasing max depth
        #fit it
        rf.fit(x_train, y_train)
        #transform it
        train_acc = rf.score(x_train, y_train)
        y_pred = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
    
        
        #evaluate on my validate data
        val_acc = rf.score(x_validate, y_validate)
        y_vpred = rf.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        scores_all.append([x, train_acc, val_acc]) 

    #create a dataframe from the scores_all list of lists
    df_scores = pd.DataFrame(scores_all)
    # add column names to the dataframe
    df_scores.columns = ['max_depth', 'train_accuracy', 'validate_accuracy']
    # sort the dataframe by the difference between train and validate
    df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
    print(df_scores.sort_values(by='delta', ascending=False))

#This function runs through a range of max_depth values and returns the accuracy scores for each max_depth value for logistic regression
def auto_lo_regress_scores(x_train, y_train, x_validate, y_validate, max_depth=range(5,20)):
        scores_all = []

        for x in max_depth:
        
        #make it
            logit = LogisticRegression() #increases the sample leaf whild decreasing max depth
                #fit it
            logit.fit(x_train, y_train)
                #transform it
            train_acc = logit.score(x_train, y_train)
            y_pred = logit.predict(x_train)
            #conf = confusion_matrix(y_train, y_pred) 
                
            #evaluate on my validate data
            val_acc = logit.score(x_validate, y_validate)
            y_vpred = logit.predict(x_validate)
            #conf = confusion_matrix(y_validate, y_vpred)

            

            scores_all.append([x, train_acc, val_acc])
        #create a dataframe from the scores_all list of lists
        df_scores = pd.DataFrame(scores_all)
        # add column names to the dataframe
        df_scores.columns = ['max_depth', 'train_accuracy', 'validate_accuracy']
        # sort the dataframe by the difference between train and validate
        df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
        print(df_scores.sort_values(by='delta', ascending=False))        

#random forest with entropy

def auto_random_forest_entropy_scores(x_train, y_train, x_validate, y_validate, max_depth=range(5,20)):
    scores_all = []

    for x in max_depth:
        
        #make it
        rf = RandomForestClassifier(criterion="entropy",random_state=123, max_depth=x) #increases the sample leaf whild decreasing max depth
        #fit it
        rf.fit(x_train, y_train)
        #transform it
        train_acc = rf.score(x_train, y_train)
        y_pred = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
    
        
        #evaluate on my validate data
        val_acc = rf.score(x_validate, y_validate)
        y_vpred = rf.predict(x_validate)
        #conf = confusion_matrix(y_validate, y_vpred)

        scores_all.append([x, train_acc, val_acc]) 

    #create a dataframe from the scores_all list of lists
    df_scores = pd.DataFrame(scores_all)
    # add column names to the dataframe
    df_scores.columns = ['max_depth', 'train_accuracy', 'validate_accuracy']
    # sort the dataframe by the difference between train and validate
    df_scores['delta'] = df_scores.train_accuracy - df_scores.validate_accuracy
    print(df_scores.sort_values(by='delta', ascending=False))

#function to test the best max_depth for the random forest model with entropy app predictor only
def auto_random_trees_test_with_entropy(x_test, y_test, x_train, y_train):
    #make it
    rf = RandomForestClassifier(criterion="entropy", max_depth=18, random_state=123) #increases the sample leaf whild decreasing max depth
    #fit it
    rf.fit(x_train, y_train)
        #transform it
        #train_acc = rf.score(x_train, y_train)
    y_pred_train = rf.predict(x_train)
        #conf = confusion_matrix(y_train, y_pred)
        #transform it
    test_acc = rf.score(x_test, y_test)
        
    y_pred = rf.predict(x_test)
        #conf = confusion_matrix(y_test, y_pred)
    
    print(f"\n------------------------ Test Model with depth of {18} Scores------------------------------")
        #plot_confusion_matrix(rf, x_test, y_test)
        #plt.show()
    print(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))
    print("------------ Metrics ----------")
        #print(print_cm_metrics(conf)) 

    print(f"Accuracy is {test_acc}")
    #plots the predicted viral apps against the actual viral apps
    plt.figure(figsize=(16,8))

    plt.hist(y_train, color='blue', alpha=.5, label="Actual App Viral")
    plt.hist(y_pred_train, color='red', alpha=.5, label="Model: Random Forest")
       #plt.hist(y_validate.G3_pred_lars, color='purple', alpha=.5, label="Model: Lasso Lars")
        #plt.hist(y_validate.G3_pred_glm, color='yellow', alpha=.5, label="Model: TweedieRegressor")
        #plt.hist(y_validate.G3_pred_lm2, color='green', alpha=.5, label="Model 2nd degree Polynomial")

    plt.xlabel("Viral")
    plt.ylabel("Number of Apps")
    plt.title("Comparing the Distribution of Viral Apps to Distributions of Predicted Viral Apps for the Top Model")
    plt.legend()
    plt.show()

#GradientBoostingClassifier
def auto_gradient_boosting_classifier(X_train, y_train, X_validate, y_validate):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=123).fit(X_train, y_train)
    print(f"Train score: {clf.score(X_train, y_train)}")
    print(f"Validate score: {clf.score(X_validate, y_validate)}")
    y_pred = clf.predict(X_validate)
    print(pd.DataFrame(classification_report(y_validate, y_pred, output_dict=True)))    

#A function for a random forest of random forests
def random_forest_voting_classifier(X_train, y_train, X_validate, y_validate):
    # Random Forest Classifier Ensemble 
    clf1 = RandomForestClassifier(criterion="entropy",random_state=123, max_depth=18)
    clf2 = RandomForestClassifier(criterion="entropy",random_state=999, max_depth=18)
    clf3 = RandomForestClassifier(criterion="entropy",random_state=789, max_depth=18)

    eclf = VotingClassifier(estimators=[('dt1', clf1), ('dt2', clf2),
                                        ('dt3', clf3)],
                            voting='hard')#, weights=[1, 1, 1])

    clf1 = clf1.fit(X_train, y_train)
    clf2 = clf2.fit(X_train, y_train)
    clf3 = clf3.fit(X_train, y_train)

    eclf = eclf.fit(X_train, y_train)


    print(f"DT1: {clf1.score(X_train, y_train)}")
    print(f"DT2: {clf2.score(X_train, y_train)}")
    print(f"DT3: {clf3.score(X_train, y_train)}")

    print(f"Ensemble: {eclf.score(X_train, y_train)}")
    print("---------------------------------------------")
    print(f"DT1: {clf1.score(X_validate, y_validate)}")
    print(f"DT2: {clf2.score(X_validate, y_validate)}")
    print(f"DT3: {clf3.score(X_validate, y_validate)}")

    print(f"Ensemble: {eclf.score(X_validate, y_validate)}")
    print("---------------------------------------------")
    y_vpred = eclf.predict(X_validate)
    print(pd.DataFrame(classification_report(y_validate, y_vpred, output_dict=True)))

#A function for a random forest of random forests test 
def random_forest_voting_classifier_test(X_train, y_train, X_test, y_test):
    # Random Forest Classifier Ensemble test
    clf1 = RandomForestClassifier(criterion="entropy",random_state=123, max_depth=18)
    clf2 = RandomForestClassifier(criterion="entropy",random_state=999, max_depth=18)
    clf3 = RandomForestClassifier(criterion="entropy",random_state=789, max_depth=18)

    eclf = VotingClassifier(estimators=[('dt1', clf1), ('dt2', clf2),
                                        ('dt3', clf3)],
                            voting='hard')#, weights=[1, 1, 1])

    clf1 = clf1.fit(X_train, y_train)
    clf2 = clf2.fit(X_train, y_train)
    clf3 = clf3.fit(X_train, y_train)

    eclf = eclf.fit(X_train, y_train)
    y_pred = eclf.predict(X_train)


    print(f"DT1: {clf1.score(X_test, y_test)}")
    print(f"DT2: {clf2.score(X_test, y_test)}")
    print(f"DT3: {clf3.score(X_test, y_test)}")

    print(f"Ensemble: {eclf.score(X_test, y_test)}")
    print("---------------------------------------------")
    y_vpred = eclf.predict(X_test)
    print(pd.DataFrame(classification_report(y_test, y_vpred, output_dict=True)))

    #plots the predicted viral apps against the actual viral apps
    plt.figure(figsize=(16,8))

    plt.hist(y_train, color='blue', alpha=.5, label="Actual App Viral")
    plt.hist(y_pred, color='red', alpha=.5, label="Model: Random Forest")
       #plt.hist(y_validate.G3_pred_lars, color='purple', alpha=.5, label="Model: Lasso Lars")
        #plt.hist(y_validate.G3_pred_glm, color='yellow', alpha=.5, label="Model: TweedieRegressor")
        #plt.hist(y_validate.G3_pred_lm2, color='green', alpha=.5, label="Model 2nd degree Polynomial")

    plt.xlabel("Viral")
    plt.ylabel("Number of Apps")
    plt.title("Comparing the Distribution of Viral Apps to Distributions of Predicted Viral Apps for the Top Model")
    plt.legend()
    plt.show()


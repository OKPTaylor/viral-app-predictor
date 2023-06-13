# Viral App Predictor 

# Project Description
 
A project designed to use classification ML modeling to predict the likelihood that an app makes it to 1 million downloads.
 
# Project Goal
 
* Find key features that drive app downloads.
* Create several classification models for the purposes of accurately predicting the likelihood an application will be downloaded 1 million or more times if it has at least 1000 downloads for applications created in 2020.
* Test the effectiveness of an ensemble style random forest model against traditional classification models. 
* Display the results of model outcomes via visualizations. 
 
# Initial Thoughts
 
It will be extremely difficult to predict the likelihood of an app’s viral success but the predictions will be more accurate with an ensemble of random forest models. 
 
# The Plan
 
* Acquire data:
    * get the Google-Platore.csv from Kaggle.com
    * Convert csv to a dataframe
 
* Prepare data:
   * Look at the data:
		* nulls
		* value counts
		* data types
		* numerical/categorical columns
		* names of columns
            * related columns
        * no outliers removed, viral apps are outliers
 
* Explore data:
   * Answer the following initial questions:
       1. Does the number of days since an app's release date affect if a million+ downloads a reached (viral)?
       2. Does editors choice and/or rating affect viral? 
       3. Does content rating affect viral?
       4. What are the best features to use?
       
* Model data:
    * 3 modeling iterations using different features and model types
    * 5 different classification models used 
        * K-Nearest Neighbors
        * Random Forest
        * Gradient Boosting Classifier
        * Random Forest of Random Forest (an ensemble model of random forest models)
        * Logistic Regression
    * 3 different visualizations, 1 for eacch iteration model test 

* Conclusions:
	* Identify features that drive downloads
    * Develop a model that beats baseline
    * Improve accuracy through using ensemble modeling  

# Data Dictionary

| Feature | Definition (measurement)|
|:--------|:-----------|
|App Name| Name of the app|
|App Id| app id|
|Category| Category the app is in (selected by creator)| 
|Rating| Quality rating given by users out of max score of 5| 
|Rating Count| Total count of the user rating| 
|Installs| Total number of installs the app has| 
|Minimum Installs| the minimum installs the app has|
|Maximum Installs| the maximum installs the app has| 
|Free| If the app is free or not|
|Price| The price of the app|
|Currency| The currency of the cost of the app|
|Size| Size of the app in kb|
|Minimum Android| The minimum Android OS needed to run the app|
|Developer Website| The website address for the developer|
|Developer Email| The Email address of the developer|
|Released| The date the app was released| 
|Privacy Policy| The link to the Privacy Policy|
|Last Updated| The date of the app’s last update|
|Content Rating| The content rating of the app|
|Ad Supported| If the app has built in ads|
|In app purchases| If the app has in app purchases| 
|Editor Choice| If the app has an editor choice award|
|Viral| If the app has 1 million or more downloads (Target Variable)|


# Steps to Reproduce
1) Clone this repo
2) Go to https://www.kaggle.com/datasets/gauthamp10/google-playstore-appsriate  
3) Download Google-Playstore.csv to the appropriate directory (do not change the file name)
4) Run notebook
 
# Takeaways and Conclusions<br>

* **The number of days an app has been out does not significantly affect it’s viral chances**
    * Time is not a predictor of app success
* **Quality rating does correlate to app success**
    * Editors choice has a significant positive correlation with the likelihood of an app going viral 
    * User rating has a small but positive impact on going viral 
* **Content rating has a very significant positive effect on an app going viral**
    * Casting as wide of a net as possible is key to an app’s acceptance 
* **There is a high correlation between in-app purchases and going viral**     
        
* **Modeling**

* Throughout 1st and 2nd iterations most models failed to beat the baseline of 97.96%. During the 2nd iteration a random forest model with a max  depth of 18 and with criterion set to “entropy” beat the base line with a accuracy score of 97.98% overall accuracy and with precision score of 77%.

* On the 3rd iteration the ensemble of random forests (random forest of random forests) beat baseline with an accuracy score of 97.98% with an precision score of 75%. 

* After much experimentation, random forest remains one of the top performing models with an accuracy score of 97.98% (beating the baseline of 97.96%) with a relatively high precision score of 77% even beating the random forest of random forests precision score by 2%. While not performing as well as the vanilla random forest model the random forest of random forests can be an useful tool in future projects.

# Recommendation
* In order to fully capture the variables that go into making an app a viral success, more data is needed on each app’s human network, i.e. grassroots marketing campaigns, social media campaigns, crowd-funding, etc.  There is more that makes a popular app than file-size and editor’s choice awards. 


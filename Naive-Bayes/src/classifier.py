import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def email_spam_classifier(training_file, testing_file):
    
    # --- training phase ---
    df = pd.read_csv(training_file, sep='\n', header=None, names=['message'])
    df['label'] = df['message'].str.split(' ').str[0]
    # print(df.head())

    # 0 is ham, 1 is spam
    # print(df['label'].value_counts())

    df_x = df["message"] #all email content
    df_y = df["label"]   #spam/ham label in 1/0 as int

    
    #split the 4000 (.8) emails for training, 1000 (.2) for validating
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size = .2, random_state = 42)

 
    
    #transform words into int
    cv = CountVectorizer()
    X_traincv = cv.fit_transform(X_train)
    X_testcv = cv.transform(X_test)
    #produce an int array from y_train
    y_train = y_train.astype(int)
    
    #call the classifier from sklearn library
    mnb = MultinomialNB()
    #train the model
    mnb.fit(X_traincv, y_train)
    #validating the model
    pred = mnb.predict(X_testcv)
    
    #produce an int array from y_train
    y_test = np.array(y_test).astype(int) 

    count=0
    for i in range(len(pred)):
        if pred[i] == y_test[i]: #compare validation predicion vs y_test reuslt
            count=count+1
    print("The training's accuracy is: ",count/len(pred)) #calculate the accuracy of the classifier's prediction

    # --- testing phase ---
    test_raw_data = pd.read_csv(testing_file, sep='\n', header=None, names=['message_test'])
    test_raw_data['label_test'] = test_raw_data['message_test'].str.split(' ').str[0]
    test_X = test_raw_data["message_test"] #all email content
    test_y = test_raw_data["label_test"]   #spam/ham label in 1/0 format
    test_X_cv = cv.transform(test_X)
    pred_test = mnb.predict(test_X_cv)
    test_y = np.array(test_y).astype(int)

    count_test=0
    for i in range(len(pred_test)):
        if pred_test[i] == test_y[i]: #compare validation predicion vs y_test reuslt
            count_test=count_test+1
    print("The testing's accuracy is: ",count_test/len(pred_test)) #calculate the accuracy of the classifier's prediction
    pass

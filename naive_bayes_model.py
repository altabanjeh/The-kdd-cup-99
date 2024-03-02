import time
import sklearn
start_time = time.time()  # Get the current time in seconds since the epoch
import data_preprocessing as dp
def naivebayesmodel():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    nb_classifier = GaussianNB()
    nb_classifier.fit(dp.x_train, dp.y_train)
    predictions = nb_classifier.predict(dp.x_test)
    return accuracy_score(dp.y_test, predictions)


    accuracy= accuracy_score(dp.y_test, predictions)

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score

    MeanAbsoluteError = round(mean_absolute_error(dp.y_test, predictions), 2)
    MeanSquaredError = round(mean_squared_error(dp.y_test, predictions),2)
    MedianAbsoluteError = round(median_absolute_error(dp.y_test, predictions),2)
    ExplainVarianceScore = round(explained_variance_score(dp.y_test, predictions),2)
    R2Score  = round(r2_score(dp.y_test, predictions),2)

    print(accuracy)
    end_time = time.time()  # Get the current time again

    elapsed_time = end_time - start_time  # Calculate the difference to find elapsed time

    t1 =elapsed_time/60 
    if t1 <1 :
        t1 = elapsed_time
        print("Elapsed time:", t1, "second ")
    else:
        print("Elapsed time:", t1, "minutes")
        
    
    return (accuracy,MeanAbsoluteError,MeanSquaredError,MedianAbsoluteError,ExplainVarianceScore,R2Score,t1)


RandomForestmodelaccuracy,MeanAbsoluteError,MeanSquaredError,MedianAbsoluteError,ExplainVarianceScore,R2Score,t1 = naivebayesmodel()


if __name__== '__naivebayesmodel__':
    naivebayesmodel()
    









 
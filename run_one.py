'''

written by :  RAED AL TABANJEH 
written on : 20/2/2024
edit on    : 27/2/2024
*******************************
this program for data_repreprocessing the data from KDD CUP 1999
this program will make the user chooses which model the user want to run 
 
    

'''
import time
import data_preprocessing
# from SCV_MODEL import SCV_model
# from DecisionTree_model import DecisionTreeModel
# from RandomForest_model import RandomForestModel
# from KNeighborsClassifier_model import KNeighborsClassifierModel
# from LogisticRegression_model import LogisticRegressionModel
# from naive_bayes_model import NaiveBayesModel

print("Welcome to KDD 1999 data preprocessing program\n")
print("Available models:\n")
print("SCV: S\n")
print("Decision Tree: DT\n")
print("Random Forest: RF\n")
print("KNeighbors Classifier: KN\n")
print("Logistic Regression: LR\n")
print("Naive Bayes: NB\n")

start_time = time.time()

model_input = input("Please enter the model you want to run: ").upper()

try:
    if model_input == "S":
        import SCVMODEL
        model = SCVMODEL()
    elif model_input == "DT":
        import DecisionTreeModel
        model = DecisionTreeModel()
    elif model_input == "RF":
        import RandomForestModel
        model = RandomForestModel()
    elif model_input == "KN":
        import KNeighborsClassifierModel
        model = KNeighborsClassifierModel()
    elif model_input == "LR":
        import LogisticRegressionModel
        model = LogisticRegressionModel()
    elif model_input == "NB":
        import NaiveBayesModel
        model = NaiveBayesModel()
    else:
        raise ValueError("Invalid model name entered")

    model.run()

except ModuleNotFoundError as e:
    print("Module not found error:", e)
except ValueError as e:
    print("Invalid input error:", e)
except Exception as e:
    print("An error occurred:", e)

end_time = time.time()
elapsed_time = end_time - start_time

if elapsed_time < 60:
    print("Elapsed time:", elapsed_time, "seconds")
else:
    print("Elapsed time:", elapsed_time / 60, "minutes")

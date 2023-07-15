import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


def get_performance(predictions, y_test, labels=[1, 0]):
    
    accuracy = accuracy_score(y_test, predictions) 
    precision = precision_score(y_test, predictions, labels = labels) 
    recall = recall_score(y_test, predictions, labels = labels) 
    f1 = f1_score(y_test, predictions, labels = labels) 
    
    report = classification_report(y_test, predictions) 
    
    cm = confusion_matrix(y_test, predictions, labels = labels) 
    cm_as_dataframe = pd.DataFrame(data = cm, index = labels, columns = labels)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1

def plot_roc(model, y_test, features):
    
    y_pred_proba = model.predict_proba(features)[:, 1] 
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba) 

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc
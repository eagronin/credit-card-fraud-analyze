# Analysis

This section constructs and evaluates several models detecting fraud in credit card transactions. The analysis focuses on selecting appropriate model evaluation metrics in the case of imbalanced classes (one class represents only a small fraction of the observations). Indeed, with only 1.6% transactions that are fraudulent, a classifier that predicts every transaction to be not fradulent will achieve 98.4% accuracy.  However, such a classifier is valueless.  Therefore, in the cases when classes are imbalanced, metrics other than accuracy should be considered.  These metrics include precision, recall and a combination of these two metrics (F2).  

Data preparation for the analysis is described in the [previous section](https://eagronin.github.io/credit-card-fraud-prepare/).

Results and visualizatons are presented in the [next section](https://eagronin.github.io/credit-card-fraud-report/).

First, we train a dummy classifier that classifies everything as the majority class of the training data (i.e., all the transactions are not fraudulent):

```python
def answer_two():
    
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    accuracy = dummy_majority.score(X_test, y_test)
    
    return accuracy
```

The function indeed returns an accuracy score of over 98%, as discussed above.  At the same time, recall (or the fraction of true positive predictions) is 0%, because the model is not designed to classify any transactions as fraudulent. Therefore, despite the high accuracy score, the model performs poorly.

Next, we train a support vector classifier (SVC) using the default parameters:

```python
def answer_three():

    svm = SVC().fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return (accuracy, recall, precision)
```

The accuracy, recall and precision are now 0.995, 0.700 and 0.965, respectively.  An increase in the recall from zero to 0.700 indicates that SVC performs substantially better than a simple majority class rule.

The following code outpust the confusion matrix:

```python
def answer_four():
    
    svm = SVC().fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    
    return confusion

print(answer_four())
```

The confusion matrix is as follows:

```
[[5342    2]
 [  24   56]]
```

We now train a logisitic regression classifier with default parameters.  For this classifier we then create a precision recall curve and a ROC curve using the test data.  A precision recall curve shows the tradeoff between recall and precision, while an ROC curve measures the cost in terms of the false positive when the number of true positives increases.

The code for plotting the precision recall curve and ROC curve is shown below:

```python
def answer_five():
        
    plt.clf()
    plt.cla()
    plt.close()
    
    lr = LogisticRegression().fit(X_train, y_train)
    y_score = lr.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(10,10))
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.xlabel('Precision', labelpad=15, fontsize = 12)
    plt.ylabel('Recall', labelpad=10, fontsize = 12)
    plt.title('Credit card fraud data analysis\nPrecision-Recall Curve\n', fontsize = 18, fontname = 'Comic Sans MS', fontweight = 'bold', alpha=1)
    plt.axes().set_aspect('equal')
    pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/precision_recall.png')
    plt.show()
    prec_rec = pd.DataFrame({'precision': precision, 'recall': recall})
    rcl = prec_rec['recall'][prec_rec.precision >= .079]
    #print('recall = ', rcl.iloc[0:10])
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,10))
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', labelpad=15, fontsize = 12)
    plt.ylabel('True Positive Rate', labelpad=10, fontsize = 12)
    plt.title('Credit card fraud data analysis\nROC curve\n', fontsize = 18, fontname = 'Comic Sans MS', fontweight = 'bold', alpha=1)
    plt.legend(loc='lower right', fontsize=12)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/roc.png')
    plt.show()
    roc = pd.DataFrame({'tpr': tpr, 'fpr': fpr})
    tr_pos_r = roc['tpr'][roc.fpr >= 0.16]
    #print('tpr = ', tr_pos_r.iloc[0:10])
        
    return (rcl.iloc[0], tr_pos_r.iloc[0])
```

As an example, the function above returns the recall of 0.81 that corresponds to the precision of 0.95 on the precision recall curve.  Similarly, it returns the true positive rate (which is another name for recall) of 0.95 that corresponds to the false positive rate of 0.16 on the ROC curve.

The visualizations of the precision recall curve and ROC curve are shown in the [next section](https://eagronin.github.io/credit-card-fraud-analyze/).

Finally, we perform a grid search over the parameters for a Logisitic Regression classifier, in order to select the best parameters to optimize performace without overfitting.  We use recall for scoring with the default 3-fold cross validation, and impose regularization penalty (both L1 and L2) for the values of C (the inverse of the regularization penalty) in the range from 0.01 to 1.

```python
def answer_six():    

    n = 20
    C_range = np.linspace(0.01, 1, n)   # yticklabels=[0.01, 0.1, 1, 10, 100]
    penalty = ['l1', 'l2']
    grid_values = dict(penalty = penalty, C = C_range)

    lr = LogisticRegression()
    grid_recall = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    grid_recall.fit(X_train, y_train)
    
    scores = grid_recall.cv_results_['mean_test_score']
    scores = scores.reshape(n,2)
    
    best_params = grid_recall.best_params_
    
    return (scores, best_params)

scores = answer_six()
best_params = scores[1]
scores = scores[0]
```

The code below outputs the best parameters obtained in the grid search:

```python
print('Best parameters obtained in the grid search:')
print('Penalty = ', best_params['penalty']) 
print('C = {:.3}'.format(best_params['C']))
```

The best parameters are as follows:

```
Best parameters obtained in the grid search:
Penalty =  l2
C = 0.0621
```

The following function visualizes the results from the grid search:

```python
def GridSearch_Heatmap(scores):

    n = 20
    C_range = np.linspace(0.01, 1, n) 

    for i in range(0, n-1):
        C_range[i] = round(C_range[i], 3)

    plt.figure(figsize=(10,10))
    sns.heatmap(scores.reshape(n,2), xticklabels=['L1','L2'], yticklabels=C_range)
    plt.title('Heatmap of the Recall\nas a Funcation of C and Penalty\n', fontsize = 18, fontname = 'Comic Sans MS', fontweight = 'bold', alpha=1)
    plt.xlabel('Penalty', labelpad = 15, fontsize = 12)
    plt.ylabel('C', labelpad = 10, fontsize = 12)
    plt.yticks(rotation=0)
    pylab.savefig('/Users/eagronin/Documents/Data Science/Portfolio/Project Output/heat_map.png');

GridSearch_Heatmap(scores)
```

Below is the code that fits the logistic regression using the best paramters obtained in the grid search, and outputs the evaluation scores.

```python
lr = LogisticRegression(penalty = best_params['penalty'], C = best_params['C']).fit(X_train, y_train)
y_score = lr.decision_function(X_test)
y_pred = lr.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
accuracy = lr.score(X_test, y_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('\nArea under ROC: {:.2}'.format(roc_auc))
print('Accuracy      : {:.2}'.format(accuracy))
print('Precision     : {:.2}'.format(precision))
print('Recall        : {:.2}'.format(recall))
```

The evaluation scores are as follows:

```
Area under ROC: 0.97
Accuracy      : 1.0
Precision     : 0.97
Recall        : 0.79
```

Next step: [Results](https://eagronin.github.io/credit-card-fraud-report/)

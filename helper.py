# data analysis & data cleaning
import pandas as pd
import numpy as np
from random                     import choices as rchoice
from random                     import sample as rs

# statistics
from scipy                      import stats

# metrics
from sklearn.metrics            import roc_auc_score, f1_score
from sklearn.metrics            import accuracy_score, cohen_kappa_score, precision_score
from sklearn.metrics            import recall_score, brier_score_loss
from sklearn.metrics            import confusion_matrix

# calibration
from sklearn.calibration        import CalibratedClassifierCV, calibration_curve

# machine learning model 
from xgboost                    import XGBClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.linear_model       import LogisticRegression
from lightgbm                   import LGBMClassifier
from imblearn.ensemble          import BalancedRandomForestClassifier
from catboost                   import CatBoostClassifier

# data visualization
import seaborn as sns
from matplotlib                 import pyplot as plt

# cross validation
from sklearn.model_selection    import cross_val_score

class helper_functions(object):

    def __init__(self):
        pass


    def cramer_v(self, x, y):
    '''Function that returns Cramér's V values from two variables x and y (arrays)'''
    # 
    cm = pd.crosstab(x, y)
    # calculate chi-square chi2
    chi2 = stats.chi2_contingency(cm)[0]
    n = cm.sum().sum()
    r, k = cm.shape
    phi = chi2/n
    phibiascorrect = max(0, phi - (k-1)*(r-1)/(n-1))
    k_denominator = k - (k-1)**2 / (n-1)
    r_denominator = r - (r-1)**2 / (n-1) 
    denominator = min(k_denominator-1, r_denominator-1)
    # return cramér's V values
    return np.sqrt(phibiascorrect / denominator)


    def IQR(self, data):
        '''Calculate the lower and upper fence of a variable's boxplot'''
        # IQR weight
        Q1 = np.quantile(data, .25)
        Q3 = np.quantile(data, .75)
        IQR = Q3 - Q1

        # calculate lower fence and using its value to eliminate outliers
        upper_fence = Q3 + (1.5 * IQR)
        lower_fence = Q1 - (1.5 * IQR)
        
        return print('For variable {}, upper fence is {} and lower fence is {}.'.format(var, upper_fence, lower_fence))


    def get_descriptive_statistics(self, data):
        '''Get descriptive statistics of a dataset and returns a dataframe'''
        
        # 1st moment (mean)
        ct_mean = pd.DataFrame(data.apply( np.mean )).T
        # median
        ct_median = pd.DataFrame(data.apply( np.median )).T

        #### Dispersion
        # 2nd moment (variance)
        d_var = pd.DataFrame(data.apply( np.var)).T
        # Standard Deviation
        d_std = pd.DataFrame(data.apply( np.std)).T
        # min
        d_min = pd.DataFrame(data.apply(min)).T
        # max
        d_max = pd.DataFrame(data.apply(max)).T
        # range
        d_range = pd.DataFrame(data.apply(lambda x: x.max() - x.min())).T
        # 3rd moment (Skew)
        d_sk = pd.DataFrame(data.apply(lambda x: x.skew())).T
        # 4th moment (Kurtosis)
        d_kurt = pd.DataFrame(data.apply(lambda x: x.kurtosis())).T
        # Q1 quantile
        d_q1 = pd.DataFrame(data.apply(lambda x: np.quantile(x, .25))).T
        # Q3 quantile
        d_q3 = pd.DataFrame(data.apply(lambda x: np.quantile(x, .75))).T

        # concatenate
        m = pd.concat([d_min, d_max, d_range, ct_mean, d_q1, ct_median, d_q3, d_std, d_sk, d_kurt]).T.reset_index()
        m.columns = ['attributes', 'min', 'max','range','mean','25%', '50%','75%','std', 'skew', 'kurtosis']
        
        return m


    def feature_importance(self, n_rows, n_cols, X_train, y_train, X_valid, y_valid):
          
        # train classifiers 
        lr = LogisticRegression(max_iter=100, random_state=42)
        lr.fit(X_train, y_train)
        lr_prob = lr.predict_proba(X_valid)
        rfc = RandomForestClassifier(n_jobs=2, random_state=42)
        rfc.fit(X_train, y_train)
        rfc_prob = rfc.predict_proba(X_valid)
        brfc = BalancedRandomForestClassifier(random_state=42)
        brfc.fit(X_train, y_train)
        brfc_prob = brfc.predict_proba(X_valid)
        cb = CatBoostClassifier(random_state=42, verbose=False)
        cb.fit(X_train, y_train)
        cb_prob = cb.predict_proba(X_valid)
        xgb = XGBClassifier(random_state=42)
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_valid)
        lgbm = LGBMClassifier(random_state=42, n_jobs=-1)
        lgbm.fit(X_train, y_train)
        lgbm_prob = lgbm.predict_proba(X_valid)
        
        feat_importance_list = [lr.coef_[0], rfc.feature_importances_, 
                            brfc.feature_importances_, cb.feature_importances_,
                        xgb.feature_importances_, lgbm.feature_importances_]
        model_name = ['Logistic Regression', 'Random Forest Classifier', 
                    'Balanced Random Forest Classifier','CatBoost Classifier',
                    'XGB Classifier','LGBM Classifier']

        # generate feature importance plots
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 20))
        sns.set(font_scale=1.5)
        for feature, name, n, ax in zip(feat_importance_list, model_name, list(range(n_rows*n_cols)), ax.flatten()):
            # get feature importance
            importance = feature
            
            # create dataframe
            df_imp = pd.DataFrame()
            
            # calculate importance of each variable
            df_imp['importance'] = pd.Series(importance, index=list(X_train.columns))
            
            # transform dataframe 
            long_df = pd.melt(df_imp.T)
            
            # plot barplot
            plt.subplot(n_rows, n_cols, n+1)
            sns.barplot(y = long_df.variable, x = long_df.value, order = long_df.sort_values('value',ascending = False)['variable'].to_list());
            plt.title(f'{name}');

        # adjusts subplot
        plt.tight_layout()

        # displays the plot
        plt.show()


    def ml_metrics(self, model_name, y_valid, y_hat, df_prob):
    '''Calculates the model performance and display metrics as a pandas dataframe.'''
        f1 = f1_score(y_valid, y_hat)
        accuracy = accuracy_score(y_valid, y_hat)
        kappa = cohen_kappa_score(y_valid, y_hat)
        roc_auc = roc_auc_score(y_valid, df_prob)
        precision = precision_score(y_valid, y_hat)
        recall = recall_score(y_valid, y_hat)
        brier = brier_score_loss(y_valid, df_prob, pos_label=1)

        return pd.DataFrame( {'Model Name': model_name,
                            'Recall': recall,
                            'Precision': precision,
                            'F1-Score': f1,
                            'ROC-AUC': roc_auc,
                            'Accuracy': accuracy,
                            'Kappa score': kappa,
                            'Brier score': brier}, index = [0])
    
    def ml_performance(self, models, X_train, y_train, X_valid, y_valid, threshold, baseline):
    '''Calculate model performance according to a list of 
    provided classifiers, train, test dataset, and a probability threshold'''
        # create empty list to show results later on
        modeling_df = []
        for k, clf in enumerate(models):
            # print model to be trained
            print("Training " + type(clf).__name__ + "...")
            if str(clf) not in ["SVC(kernel='linear', random_state=42)","RidgeClassifier()"]:
                # fits the classifier to training data
                clf.fit(X_train, y_train)

                # predict the probabilities. This generates two-numpy arrays: 0 = prob of patient not having CVD, 1 = otherwise
                clf_prob = clf.predict_proba(X_valid)

                # data-framing the array 1 (only probabilities of having CVD), and naming the column as 'prob'
                df_prob = pd.DataFrame(clf_prob[:, 1], columns=['prob'])

                # apply threshold to dataframe 'probs'. If probabilities are higher than the threshold, we replace the probability value by 1 (has CVD) or 0 otherwise.
                y_hat = df_prob['prob'].apply(lambda x: 1 if x > threshold else 0)

                # calculate metrics and add to empty list
                modeling_result = self.ml_metrics(type(clf).__name__, y_valid, y_hat, df_prob)

                # add metrics in an empty list
                modeling_df.append(modeling_result)
                
            if str(clf) in ["SVC(kernel='linear', random_state=42)","RidgeClassifier()"]:
                # fits the classifier to training data
                clf.fit(X_train, y_train)

                # apply threshold to dataframe 'probs'. If probabilities are higher than the threshold, we replace the probability value by 1 (has CVD) or 0 otherwise.
                y_hat = clf.predict(X_valid)

                # calculate metrics and add to empty list
                modeling_result = self.ml_metrics(type(clf).__name__, y_valid, y_hat, df_prob)

                # add metrics in an empty list
                modeling_df.append(modeling_result)          
            
        if baseline == 'yes':
            ### add baseline model
            # Generating random probability numbers from 0 to 1
            baseline_yhat = rchoice(population = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], k = len(y_hat))

            # data-framing the array 1 (only probabilities of having CVD), and naming the column as 'prob'
            df_prob = pd.DataFrame(baseline_yhat, columns=['prob'])

            # apply threshold to dataframe 'probs'. 
            y_hat = df_prob['prob'].apply(lambda x: 1 if x > threshold else 0)
            baseline_model = self.ml_metrics('Baseline Model (Guess)', y_valid, y_hat, df_prob)

            # add metric in an empty list & concatenate dataframe
            modeling_df.append(baseline_model)
            final_df = pd.concat(modeling_df)
        else:
            # concatenate all classifier performances into a unique dataframe
            final_df = pd.concat(modeling_df)

        # return dataframe sorted by the f1-score
        return final_df.sort_values('F1-Score', ascending = False)  
        
    def single_confusion_matrix(self, y_valid, y_pred, model, qualifier=""):
        # calculates confusion matrix
        cm = confusion_matrix(y_valid, y_pred)

        # plots confusion matrix as heatmap
        sns.set(font_scale=1)
        ax = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                        square=True, annot_kws={"size": 16}, cbar_kws={"shrink": 0.4})

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        ax.title.set_text(type(model).__name__ + ' ' + str(qualifier))

    def multiple_confusion_matrices(self, n_rows, n_cols, X_train, y_train, X_valid, y_valid, models, threshold = 0.50):
        '''Print multiple confusion matrices'''
        
        # define subplots
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 16))
        
        for clf, ax, n in zip(models, ax.flatten(), list(range(n_rows*n_cols))):
            
            # fits the classifier to training data
            clf.fit(X_train, y_train)
            
            # predict the probabilities
            clf_probs = clf.predict_proba(X_valid)

            # keeps probabilities for the positive outcome only
            probs = pd.DataFrame(clf_probs[:, 1], columns=['prob_default'])

            # applied the threshold
            y_pred = probs['prob_default'].apply(
                lambda x: 1 if x > threshold else 0)

            # plots confusion matrix as heatmap
            plt.subplot(n_rows, n_cols, n+1)
            self.single_confusion_matrix(y_valid, y_pred, model = clf)

        # adjusts subplot
        plt.tight_layout()

        # displays the plot
        plt.show()

    def cross_validation(self, models, X, y, cv, qualifier=""):
        '''Calculate cross validation and display results in a dataframe'''
        # create empty list to show results later on
        cv_df = []
        for n in models:
            prec_cv = cross_val_score(n, X, y, cv = cv, scoring='precision', n_jobs=-1)
            prec = "{:.4f} +/- %{:.4f}".format(prec_cv.mean(), prec_cv.std())
            recall_cv = cross_val_score(n, X, y, cv = cv, scoring='recall', n_jobs=-1)
            recall = "{:.4f} +/- %{:.4f}".format(recall_cv.mean(), recall_cv.std())
            f1_score_cv = cross_val_score(n, X, y, cv = cv, scoring='f1', n_jobs=-1)
            f1_score = "{:.4f} +/- %{:.4f}".format(f1_score_cv.mean(), f1_score_cv.std())
            roc_auc_cv = cross_val_score(n, X, y, cv = cv, scoring='roc_auc', n_jobs=-1)
            roc_auc = "{:.4f} +/- %{:.4f}".format(roc_auc_cv.mean(), roc_auc_cv.std())
            accuracy_cv = cross_val_score(n, X, y, cv = cv, scoring='accuracy', n_jobs=-1)
            accuracy = "{:.4f} +/- %{:.4f}".format(accuracy_cv.mean(), accuracy_cv.std())
                    
            a1 = pd.DataFrame( {'Model Name': type(n).__name__ + ' ' + str(qualifier),
                        'Precision (Avg+Std) ': prec,
                        'Recall (Avg+Std) ': recall,
                        'F1-Score (Avg+Std)': f1_score,
                        'ROC-AUC (Avg+Std)': roc_auc,
                        'Accuracy (Avg+Std)': accuracy
                        }, index = [0])
        
            # add metrics in an empty list
            cv_df.append(a1)
        # concatenate all classifier performances into a unique dataframe
        final_df = pd.concat(cv_df)
            
        # return dataframe sorted by the f1-score
        return final_df

    
    def random_search_lgbm(self, param, n_iterations, X, y):
        '''Select the best parameters for the lgbm model'''
        # allocate all results in dataframe
        final_result = pd.DataFrame(columns=['mean f1-score', 'std', 'parameters'])
        for i in range( n_iterations ):
            # choose values for parameters randomly
            hp = { k: rs( v, 1 )[0] for k, v in param.items() }

            # model
            model_random_search = LGBMClassifier( objective='binary',
                                        num_leaves = hp['num_leaves'],
                                        min_data_in_leaf=hp['min_data_in_leaf'], 
                                        learning_rate=hp['learning_rate'], 
                                        n_estimators=hp['n_estimators'], 
                                        max_depth=hp['max_depth'], 
                                        colsample_bytree=hp['colsample_bytree'],
                                        min_child_weight=hp['min_child_weight'], random_state=42, n_jobs=-1).fit(X, y)

            # define CV strategy
            sk_fold = StratifiedKFold(n_splits=10, random_state=None)
            # calculate cross validation
            cv_scores = cross_val_score(model_random_search, X, y, cv = sk_fold, scoring='f1', n_jobs=-1)

            # append cv scores in dataframe
            result = pd.DataFrame([[mean(cv_scores), std(cv_scores), hp]], 
                                columns=['mean f1-score','std','parameters'])
            final_result = pd.concat( [final_result, result] )

        return final_result.sort_values('mean f1-score', ascending = False).head(10)    

    def plot_calibration_curve(self, model, name, fig_index, X_train, y_train, X_test, y_test):
        """Plot calibration curve for est w/o and with calibration. """
        # Calibrated with isotonic calibration
        isotonic = CalibratedClassifierCV(model, cv=2, method='isotonic')

        # Calibrated with sigmoid calibration
        sigmoid = CalibratedClassifierCV(model, cv=2, method='sigmoid')

        # Logistic regression with no calibration as baseline
        lgbm = LGBMClassifier(random_state=42, n_jobs=-1)


        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        df_scores = []
        for clf, name in [(lgbm, 'LGBM'),
                        (model, name),
                        (isotonic, name + ' + Isotonic'),
                        (sigmoid, name + ' + Sigmoid')]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            prob_pos = clf.predict_proba(X_test)[:, 1]
            clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label="%s (%1.3f)" % (name, clf_score))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                    histtype="step", lw=2)
        
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve) - with brier scores')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper left", ncol=2)
        plt.tight_layout()
        plt.show()   

    def calibrated_scores(self, model, name, fig_index, X_train, y_train, X_test, y_test):
        """Plot calibration scores for est w/o and with calibration. """
        # Calibrated with isotonic calibration
        isotonic = CalibratedClassifierCV(model, cv=2, method='isotonic')

        # Calibrated with sigmoid calibration
        sigmoid = CalibratedClassifierCV(model, cv=2, method='sigmoid')

        # Logistic regression with no calibration as baseline
        lgbm = LGBMClassifier(random_state=42, n_jobs=-1)
        df_scores = []    
        for clf, name in [(lgbm, 'LGBM'),
                        (model, name),
                        (isotonic, name + ' + Isotonic'),
                        (sigmoid, name + ' + Sigmoid')]:
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            prob_pos = clf.predict_proba(X_test)[:, 1]
            clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
            a1 = pd.DataFrame( {'Model Name': name,
                            'Precision': precision_score(y_test, y_pred),
                            'Recall': recall_score(y_test, y_pred),
                            'F1-Score': f1_score(y_test, y_pred),
                            'ROC-AUC': roc_auc_score(y_test, y_pred),
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Kappa': cohen_kappa_score(y_test, y_pred),
                            'Brier Score': clf_score}, index = [0])
            df_scores.append(a1)
        # concatenate all classifier performances into a unique dataframe
        df_final = pd.concat(df_scores)
        return df_final
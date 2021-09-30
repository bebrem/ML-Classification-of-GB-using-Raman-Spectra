## Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from scipy.signal import savgol_filter
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
import matplotlib
import seaborn as sns
from umap import UMAP
from scipy import sparse
from scipy.sparse.linalg import spsolve
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

## Static Parameters
cv = 5
negative_train = 10
positive_train = 10
window_length = 61
deriv = 1
polyorder = 4
sav_gol_enable = False
pca_enable = True
fs_enable = True
lda_enable = False
baseline_enable = False
## Static Functions
# 1 - Function to convert txt to a dataframe
def txt_to_dataframe(directory,label,sav_gol_enable=sav_gol_enable):
    dataframe = pd.DataFrame()
    label_list = []
    all_data = pd.read_csv(directory,delimiter="\t")
    intensity_df = all_data["Unnamed: 3"]
    counter = 0
    for i in range(0,len(intensity_df),1015):
        if counter >= 100:
            break
        current_sample = intensity_df[i:i+1015].to_numpy().transpose()
        current_sample = current_sample[0:1010]
        if sav_gol_enable:
            current_sample = sav_gol(current_sample)
        if baseline_enable:
            current_sample = baseline_als(current_sample, lam=5, p=0.1, niter=10)
        smooth_sample = pd.DataFrame(current_sample).transpose()
        dataframe = dataframe.append(smooth_sample)
        label_list.append(label)
        counter = counter + 1
    #dataframe["label"] = label_list
    return dataframe  

# 2 - Savitzky Golay Filter
def sav_gol(X,deriv=deriv,window_length=window_length,polyorder=polyorder):
    return savgol_filter(X, window_length = window_length,polyorder = polyorder,deriv = deriv)

# 3 - Effect of the Training Data on the Performance
def corr_matrix(training_set_list,cv_accuracy):
    tsl = pd.DataFrame(training_set_list)
    reg = LassoCV()
    reg.fit(tsl, cv_accuracy)
    coef = pd.Series(reg.coef_, index = tsl.columns)
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso Model")

# 4 - Baselie Correction , lam = 5, p = 0.1
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

# 5 - t-SNE plot 
# Note: for the algorithm to operate properly, the perplexity really should be smaller than the number of points.
# Default values: n_iter = 1000, learning_rate=200, perplexity=30
def tsne_plot(x,y,perplexity,learning_rate, n_iter,labels):
    tsne = TSNE(n_components=2, random_state=0,perplexity=perplexity,learning_rate=learning_rate,n_iter=n_iter)
    projections = tsne.fit_transform(x)
    projections_dataframe = pd.DataFrame(projections)
    projections_dataframe.columns = ['a','b']
    projections_dataframe['label'] = np.array(y).reshape(len(y),1)
    g = sns.lmplot(x="a", 
               y="b", 
               data=projections_dataframe,
               fit_reg = False,
               legend = False,
               aspect = 1.5,
               height = 9,
               hue = 'label',
               scatter_kws = {"s":200, "alpha":0.3})
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 16,
            }
    plt.rc('font', **font)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='upper right', labels=labels)
    #plt.rcParams['font.size']='16'
    plt.show(g)


    
# 6 - PCA Plot
def pca_plot(X,y,n_components=4):
    pcap = PCA(n_components=n_components)
    pc = pcap.fit_transform(X)
    pc_df = pd.DataFrame(data = pc, 
        columns = ['PC1', 'PC2','PC3','PC4'])
    pc_df['Cluster'] = np.array(y).reshape(len(y),1)      
    sns.lmplot( x="PC1", y="PC2",
      data=pc_df, 
      fit_reg=False, 
      hue='Cluster', # color by cluster
      legend=True,
      scatter_kws={"s": 80}) # specify the point size

# 7 - uMAP Plot
def umap_plot(X,y,n_components=2):
    umap_2d = UMAP(n_components=n_components, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(X)
    proj_df = pd.DataFrame(data = proj_2d, 
        columns = ['PJ1', 'PJ2'])
    proj_df['Cluster'] = np.array(y).reshape(len(y),1)   
    sns.lmplot( x="PJ1", y="PJ2",
      data=proj_df, 
      fit_reg=False, 
      hue='Cluster', # color by cluster
      legend=True,
      scatter_kws={"s": 80}) # specify the point size
    
## One Hot Encoding for Covariance Analysis
gm_patient_list = os.listdir("gm/")
wm_patient_list = os.listdir("wm/")
tm_patient_list = os.listdir("tm/")
nc_patient_list = os.listdir("nc/")
complete_patient_list = gm_patient_list + wm_patient_list + tm_patient_list + nc_patient_list
enc = LabelEncoder()
enc.fit(complete_patient_list)
# How to apply: ohe_trainingset = enc.transform(np.array(training_set_list).reshape(32))


## Model Training & Test
def model_development(cv=cv,baseline_enable = baseline_enable,sav_gol_enable=sav_gol_enable,fs_enable=fs_enable,lda_enable=lda_enable,pca_enable=pca_enable,negative_train=negative_train,positive_train=positive_train,complete_patient_list=complete_patient_list):
    cv_accuracy = []
    patient_accuracy = []
    roc_auc = []
    number_of_training = negative_train + positive_train + 2
    cm_list = np.zeros((cv,2,2))
    fpr = [[0 for i in range(16)] for j in range(cv)]
    tpr = [[0 for i in range(16)] for j in range(cv)]
    threshold = [[0 for i in range(16)] for j in range(cv)]
    cv_current = 0
    test_number = len(complete_patient_list) - number_of_training 
    blindTest = np.zeros((cv,35))
    blindLabel = np.zeros((cv,35))
    testSetName = [["null" for i in range(test_number)] for j in range(cv)]
    training_set_list = [["null" for i in range(number_of_training)] for j in range(cv)]
    training_set_list_enc = [["null" for i in range(number_of_training)] for j in range(cv)]
    ## Shuffle the train-test dataset
    for i in range(cv):
    # Import the dataset
    # WM
        patient_index = 0
        patient_list = os.listdir("wm/")
        random.shuffle(patient_list)
        wmTrain = pd.DataFrame()
        wmTest = pd.DataFrame()
        index = 0
        for patient in patient_list:
            wm_directory = "wm/"+patient
            current_df = txt_to_dataframe(wm_directory,label=0,sav_gol_enable=sav_gol_enable)
            if index > negative_train:
                wmTest = wmTest.append(current_df)
            else:
                training_set_list[cv_current][patient_index]=patient
                wmTrain = wmTrain.append(current_df)
                patient_index = patient_index +1
            index = index + 1
    
        # TM
        patient_list = os.listdir("tm/")
        random.shuffle(patient_list)
        tmTrain = pd.DataFrame()
        tmTest = pd.DataFrame()
        index = 0
        for patient in patient_list:
            tm_directory = "tm/"+patient
            current_df = txt_to_dataframe(tm_directory,label=1,sav_gol_enable=sav_gol_enable)
            if index > positive_train:
                tmTest = tmTest.append(current_df)
            else:
                training_set_list[cv_current][patient_index]=patient
                tmTrain = tmTrain.append(current_df)
                patient_index = patient_index +1
            index = index + 1
            
        training_set_list_enc[cv_current] = enc.transform(np.array(training_set_list[cv_current]).reshape(number_of_training))
        train_set = wmTrain.append(tmTrain)
        y_train = np.append(np.zeros(len(wmTrain)),np.ones(len(tmTrain)),axis=0)
        test_set = wmTest.append(tmTest)
        y_test = np.append(np.zeros(len(wmTest)),np.ones(len(tmTest)),axis=0)
        
        # Normalize the data
        x_train = preprocessing.minmax_scale(train_set.T).T  
        x_test = preprocessing.minmax_scale(test_set.T).T  
    
        # Feature Scaling if enables
        if fs_enable:
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            
        # PCA if enabled
        if pca_enable:
            pca = PCA(0.95)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
        
        # PCA Loadings
        # pca_loadings = np.append(x_train,x_test,axis=0)
        # sc = StandardScaler()
        # pca_loadings = sc.fit_transform(pca_loadings)
        # pca = PCA(n_components=2)
        # pca_loadings = pca.fit_transform(pca_loadings)
        # loadings = pd.DataFrame(pca.components_.T* np.sqrt(pca.explained_variance_), columns=['PC1', 'PC2'])
        # loadings
        # ax = sns.heatmap(loadings)
        # loadings.to_csv('pca_components.csv')       
        
        
        # LDA if enabled
        if lda_enable:
            lda = LDA(n_components=1)
            x_train = lda.fit_transform(x_train,y_train)
            x_test = lda.transform(x_test)
            
            
        # # Parameter Optimization
        # param_grid = {'C': [0.1, 1, 10], 
        #               'gamma': [1, 0.1, 0.01],
        #               'kernel': ['rbf','linear','poly','sigmoid'],
        #               'decision_function_shape': ['ovo', 'ovr']} 
        # grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
        # # fitting the model for grid search
        # grid.fit(x_train, y_train)
        
        # # print best parameter after tuning
        # print(grid.best_params_)
        # # print how our model looks after hyper-parameter tuning
        # print(grid.best_estimator_)
        
        # Train SVM
        # UNCOMMENT the next line for SVM model
        #classifier = SVC(kernel = 'linear',C= 0.1, decision_function_shape= 'ovo', gamma= 1, random_state = 0) #RandomForestClassifier(random_state=0)

        # UNCOMMENT the next line for Random Forest model
        # classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
        
        # UNCOMMENT the next line for XGBoost model
        classifier = XGBClassifier( learning_rate =0.1, n_estimators=500, max_depth=7,
                                   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.6,
                                   objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=47, reg_alpha = 0.1)

        
        classifier.fit(x_train, y_train)
        yhat = classifier.predict(x_test)
        
        # Calculate the classification accuracy
        valid_accuracy = accuracy_score(y_test, yhat)
        cv_accuracy.append(valid_accuracy)  
        
        
        # Patientwise Predictions
        blind_index = 0
        blind_label = []
        valid_patient_list = os.listdir("wm/") 
        for patient in valid_patient_list:
            patient_in_training = False
            for training_patient in training_set_list[cv_current]:
                if training_patient == patient:
                    patient_in_training = True
            if not patient_in_training:
                testSetName[cv_current][blind_index] = patient
                valid_pred = []
                valid_dir = "wm/" + patient
                current_df = txt_to_dataframe(valid_dir,label=0,sav_gol_enable=sav_gol_enable)
                y_valid = np.ones(len(current_df))
                blind_label.append(0)
                X_valid = preprocessing.minmax_scale(current_df.T).T 
                if fs_enable:
                    X_valid = sc.transform(X_valid)
                if pca_enable:
                    X_valid = pca.transform(X_valid)
                if lda_enable:
                    X_valid = lda.transform(X_valid)
                valid_pred = classifier.predict(X_valid)
                valid_pred = (valid_pred > 0.5)
                valid_accuracy = metrics.accuracy_score(y_valid, valid_pred)
                blindTest[cv_current,blind_index] = valid_accuracy
                blind_index = blind_index+1
        
        valid_patient_list = os.listdir("tm/") 
        for patient in valid_patient_list:
            patient_in_training = False
            for training_patient in training_set_list[cv_current]:
                if training_patient == patient:
                    patient_in_training = True
            if not patient_in_training:
                testSetName[cv_current][blind_index] = patient
                valid_pred = []
                valid_dir = "tm/" + patient
                current_df = txt_to_dataframe(valid_dir,label=0,sav_gol_enable=sav_gol_enable)
                y_valid = np.ones(len(current_df))
                blind_label.append(1)
                X_valid = preprocessing.minmax_scale(current_df.T).T 
                if fs_enable:
                    X_valid = sc.transform(X_valid)
                if pca_enable:
                    X_valid = pca.transform(X_valid)
                if lda_enable:
                    X_valid = lda.transform(X_valid)
                valid_pred = classifier.predict(X_valid)
                valid_pred = (valid_pred > 0.5)
                valid_accuracy = metrics.accuracy_score(y_valid, valid_pred)
                blindTest[cv_current,blind_index] = valid_accuracy
                blind_index = blind_index+1                                
        
           
        # ROC Curve
        blind_label2 = np.array(blind_label)
        blindLabel[cv_current,:] = blind_label2
        fpr[cv_current], tpr[cv_current], threshold[cv_current] = roc_curve(blind_label2,blindTest[cv_current,:])
        roc_auc.append(auc(fpr[cv_current], tpr[cv_current]))
        y_pred_cm = blindTest[cv_current,:]>=0.5
        cm = confusion_matrix(blind_label2,y_pred_cm)
        cm_list[cv_current,:,:] = cm
        pt_accuracy = accuracy_score(blind_label2, y_pred_cm)
        patient_accuracy.append(pt_accuracy)
        cv_current = cv_current+1
        
    return patient_accuracy,testSetName,blindLabel,cv_accuracy,blindTest,training_set_list_enc,fpr,tpr,threshold,roc_auc,cm_list

# Run the code block and analyse the results
patient_accuracy,testSetName,blindLabel,cv_accuracy,blindTest,training_set_list_enc,fpr,tpr,threshold,roc_auc,cm_list = model_development(cv=cv,baseline_enable = baseline_enable,sav_gol_enable=sav_gol_enable,fs_enable=fs_enable,lda_enable=lda_enable,pca_enable=pca_enable,negative_train=negative_train,positive_train=positive_train,complete_patient_list=complete_patient_list)

# Uncomment the following lines for saving the results
# df = pd.DataFrame(cv_accuracy).T
# df.to_excel(excel_writer = "C:/Users/Buse/OneDrive - Koç Üniversitesi/Research\GBM Results\Algorithms\SVM\samplewise_accuracies.xlsx")

# df2 = pd.DataFrame(patient_accuracy).T
# df2.to_excel(excel_writer = "C:/Users/Buse/OneDrive - Koç Üniversitesi/Research\GBM Results\Algorithms\SVM\ptwise_accuracies.xlsx")

# df3 = pd.DataFrame(tpr).T
# df3.to_excel(excel_writer = "C:/Users/Buse/OneDrive - Koç Üniversitesi/Research\GBM Results\Algorithms\SVM\ptwise_tpr.xlsx")

# df4 = pd.DataFrame(fpr).T
# df4.to_excel(excel_writer = "C:/Users/Buse/OneDrive - Koç Üniversitesi/Research\GBM Results\Algorithms\SVM\ptwise_fpr.xlsx")

# df5 = pd.DataFrame(roc_auc).T
# df5.to_excel(excel_writer = "C:/Users/Buse/OneDrive - Koç Üniversitesi/Research\GBM Results\Algorithms\SVM\ptwise_roc_auc.xlsx")

# df6 = pd.DataFrame(threshold).T
# df6.to_excel(excel_writer = "C:/Users/Buse/OneDrive - Koç Üniversitesi/Research\GBM Results\Algorithms\SVM\ptwise_threshold.xlsx")

# ROC Curve for all CVs
plt.title('Receiver Operating Characteristic')
for ind in range(cv):
    plt.plot(fpr[ind], tpr[ind], label = 'AUC = %0.2f' % roc_auc[ind])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
plt.show()

# Threshold Opt.
ind = 1
gmeans = np.sqrt(tpr[ind] * (1-fpr[ind]))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ind][ix], gmeans[ix]))

# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--')
plt.plot(fpr[ind], tpr[ind], marker='.', label='AUC = %0.2f' % roc_auc[ind])
plt.scatter(fpr[ind][ix], tpr[ind][ix], marker='o', color='black', label='Best = %0.2f' % threshold[ind][ix])
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()

ind = 9 #3
print(cm_list[ind])
y_pred_cm = blindTest[ind]>=threshold[ind][ix]
print(confusion_matrix(blindLabel[ind,:],y_pred_cm))



# Using Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(blindLabel[ind,:],blindTest[ind,:])
# plot the roc curve for the model
no_skill = len(blindTest[ind,:][blindTest[ind,:]==1]) / len(blindTest[ind,:])
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(blindTest[ind,:][blindTest[ind,:]==1]) / len(blindTest[ind,:])
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
# show the plot
plt.show()

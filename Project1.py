
# coding: utf-8

# ## Check the data

# In[16]:


from sklearn.datasets import fetch_20newsgroups
# Refer to the offcial document of scikit-learn for detailed usages:
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
twenty_train = fetch_20newsgroups(subset='train', # choose which subset of the dataset to use; can be 'train', 'test', 'all'
                                  shuffle=True,
                                  random_state=42, # set the seed of random number generator when shuffling to make the outcome repeatable across different runs
#                                   remove=['headers'],
                                 )
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)


# In[18]:


print(type(twenty_train))
print(twenty_train.keys())


# In[19]:


twenty_train.filenames


# In[29]:


twenty_train.target


# In[23]:


twenty_train.target_names


# In[24]:


len(twenty_train.target_names)


# In[28]:


# `twenty_train.target` only contains 0 and 1
import numpy as np
np.unique(twenty_train.target)


# In[40]:


print(twenty_train.description)
print(twenty_train.target[1])
print(twenty_train.data[1])


# In[35]:


# The first document belongs to the category 'comp.sys.mac.hardware'
print(twenty_train.target_names[twenty_train.target[0]])


# In[39]:


print(len(twenty_train.data))
print(len(twenty_train.target))


# ## Question 1:
# To get started, plot a histogram of the number of training documents for each
# of the 20 categories to check if they are evenly distributed.

# In[42]:


len(twenty_train.target==1)


# In[46]:


num = [0]*20
lenth = len(twenty_train.target)
for i in range(0,lenth):
    num[twenty_train.target[i]]+=1
print(num)


# In[73]:


from matplotlib import pyplot as plt
t = range(0,20)
plt.bar(t,num,0.5)
plt.xticks(t,t)
plt.title("Distribution of documents")
plt.xlabel("Category ID")
plt.ylabel("Number of documents in the group")
plt.show()


# ## Question 2: 
# Use the following specs to extract features from the textual data:
# - Use the “english” stopwords of the CountVectorizer
# - Exclude terms that are numbers (e.g. “123”, “-45”, “6.7” etc.) 
# - Perform lemmatization with nltk.stem.wordnet.WordNetLemmatizer and pos_tag 
# - Use min_df=3 
# 
# Report the shape of the TF-IDF matrices of the train and test subsets respectively

# In[2]:


from sklearn.datasets import fetch_20newsgroups
categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = None)


# In[3]:


import numpy as np
print(np.unique(train_dataset.target))
print(train_dataset.target_names)
print(len(train_dataset.target))
print(len(test_dataset.target))
print(type(train_dataset.target))


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=3,token_pattern=r'(?u)\b[a-zA-Z]{2,}\b',stop_words='english') 
# ngram is n words together to be concidered as a phrase
# max_df = 1.0 means if the context only consists of the same word, then it will be eliminated.
# min_df = 3 (int) means if the context contains the word repeats less than 3 times, then it will be ignored.
vectorizer


# In[4]:


X_train = vectorizer.fit_transform(train_dataset.data)
print(X_train.shape)
print('-' * 20)
print(len(vectorizer.get_feature_names()))
print('-' * 20)
print(vectorizer.get_feature_names()) # not lemmatized


# In[5]:


# Lemmatization
from nltk import pos_tag,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

wnl = WordNetLemmatizer()
X_train = vectorizer.fit_transform(train_dataset.data)

# why 'ran' is considered to be NOUN in pos_tag?

# Here 2nd para of lemmatize doesn't support pos_tag. it should be transfered to wordnet.NOUN or ... to fit in

# Define a tag_trans to fit in pos in wnl.lemmatize(_,pos=_)

#The function is from Author: 
#Author：但盼风雨来_jc
#Link：https://www.jianshu.com/p/79255fe0c5b5
#Resource：简书
#简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

def tag_trans(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return 'n'

org_wordlist = vectorizer.get_feature_names()
#org_wordlist = ['I','loves','bunnies','her','she','running','ran']*3  #testset
new_wordlist = []
for word in org_wordlist:
    tag = pos_tag(word_tokenize(word))[0][1]
    pos = tag_trans(tag)
    new_word = wnl.lemmatize(word,pos)
    new_wordlist.append(new_word)

#a=tag_trans(pos_tag(word_tokenize('aaa'))[0][1])
#print(a)
#print(wnl.lemmatize('aaa',pos=a))
    
print(len(new_wordlist))
#print(new_wordlist)
print('-' * 20)

vec = CountVectorizer() # new vectorizer 
print(vec)
print('-' * 20)
base = vec.fit_transform(new_wordlist)
X_train = vec.transform(train_dataset.data)
print(X_train.shape)
print('-' * 20)
print(len(vec.get_feature_names()))
print('-' * 20)
print(vec.get_feature_names())


# In[36]:


# Lemmatization
from nltk import pos_tag,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
np.random.seed(42)
import random
random.seed(42)

wnl = WordNetLemmatizer()

stop_words = ENGLISH_STOP_WORDS

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_sent_demo(text):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(nltk.word_tokenize(text))]
def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in stop_words and not word.isdigit())

analyzer = CountVectorizer().build_analyzer()
vec = CountVectorizer(min_df=3, analyzer=stem_rmv_punc, stop_words = 'english') # new vectorizer 
print(vec)
print('-' * 20)
X_train = vec.fit_transform(train_dataset.data)
print(X_train.shape)
print('-' * 20)
print(len(vec.get_feature_names()))
print('-' * 20)
print(vec.get_feature_names())


# In[37]:


# test data
X_test = vec.transform(test_dataset.data)
print(X_test.shape)
print(X_test)


# In[7]:


# TF-IDF matrices of the train subsets
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

print(tfidf)
print('-' * 20)

X_train_tfidf = tfidf.fit_transform(X_train)
print(X_train_tfidf.shape)
print('-' * 20)
print(X_train.toarray()[:30,:30])
print('-' * 20)
print(X_train_tfidf.toarray()[:30,:30])


# In[8]:


# TF-IDF matrices of the test subsets
X_test_tfidf = tfidf.transform(X_test)
print(X_test_tfidf.shape)
print('-' * 20)
print(X_test.toarray()[:30,:30])
print('-' * 20)
print(X_test_tfidf.toarray()[:30,:30])


# ## Question 3:
# Reduce the dimensionality of the data using the methods above
# - Apply LSI to the TF-IDF matrix corresponding to the 8 categories with k = 50; so each document is mapped to a 50-dimensional vector.
# - Also reduce dimnsionality through NMF (k = 50) and compare with LSI: 
# <br>Which one is larger, the $ \mid\mid X-WH\mid\mid_F^2 $ in NMF or the $\mid\mid X-U_k\sum_kV_k^T\mid\mid_F^2$ in LSI? Why is the case?

# In[9]:


# LSI
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components = 50, random_state = 42)
X_train_lsi = svd.fit_transform(X_train_tfidf)
X_test_lsi = svd.transform(X_test_tfidf)

# show time
print(X_train_lsi.shape)
print(X_train_lsi)


# In[10]:


# NMF
from sklearn.decomposition import NMF

nmf = NMF(n_components=50, init='random',random_state=42)
W_train = nmf.fit_transform(X_train_tfidf)
W_test = nmf.transform(X_test_tfidf)

H = nmf.components_

# show time
print(W_train.shape)
print('-' * 20, 'NMF', '-'*20)
print(W_train)
print('-' * 20, 'NMF', '-'*20)
print(H.shape)


# In[11]:


# Loss of nmf

import numpy as np

err_nmf = nmf.reconstruction_err_
err = np.sum(np.array(X_train_tfidf-W_train.dot(H))**2)
err = err**0.5

print(err_nmf)
print(err)


# In[12]:


# Check attributes of the svd function

check = svd.components_.dot(np.transpose(svd.components_))
print(svd.components_.shape)
print(check)

err_lsi_test = np.sum(np.array(X_test_tfidf-X_test_lsi.dot(svd.components_))**2)
err_lsi_test = err_lsi_test**0.5

print(err_lsi_test)


# In[13]:


# Loss of lsi

err_lsi = np.sum(np.array(X_train_tfidf-X_train_lsi.dot(svd.components_))**2)
err_lsi = err_lsi**0.5
print('The loss of LSI method is: ',err_lsi)
print('The loss of NMF method is: ',err_nmf)

if err_lsi > err_nmf:
    print('So the loss of LSI is larger!')
elif err_lsi < err_nmf:
    print('So the loss of NMF is larger!')
else:
    print('Incredible!!! They are equal!')


# ## Question 4:
# Hard margin and soft margin linear SVMs:
# - Train two linear SVMs and compare:
#     * Train one SVM with $\gamma = 1000$ (hard margin), another with $\gamma = 0.0001$ (soft nargin).
#     * Plot the ROC curve, report the **confusion matrix** and calculate the **accuracy, recall, precision** and **F-1 score** of both SVM classifier. Which one performs better?
#     * What happens for the soft margin SVM? Why is the case?
#         - Does the DOC curve of the soft margin SVM look good? Does this conflict with other metrics?
# - Use cross-validation to choose $\gamma$ (use average validation accuuracy to compare):
# <br>Using a 5-fold cross-validation, find the best value of the parameter $\gamma$ in the range $\lbrace10^k\mid-3\leq k\leq3, k\in\mathbb Z\rbrace$. Again, plot the ROC curve and report the confusion matrix and calculate the **accuracy, recall precision** and **F-1 score** of this best SVM.

# ### Train SVMs and compare

# In[70]:


# data preparation and regroup

# number and class name into dict
#num = np.unique(train_dataset.target)
#name = train_dataset.target_names
#name_dic = dict(zip(name,num))
#print(name_dic)

# divide into groups
#grp1 = (name_dic.get('comp.graphics'), name_dic.get('rec.autos'))
#grp2 = (name_dic.get('comp.os.ms-windows.misc'), name_dic.get('rec.motorcycles'))
#grp3 = (name_dic.get('comp.sys.ibm.pc.hardware'), name_dic.get('rec.sport.baseball'))
#grp4 = (name_dic.get('comp.sys.mac.hardware'), name_dic.get('rec.sport.hockey'))

# regroup
def get_labels(train_dataset):
    labels = []
    new_name = ['Computer Technology','Recreational Activity']

    for label in train_dataset.target:
        name = train_dataset.target_names[label]
        if name.startswith('comp'):
            labels.append(0)
        elif name.startswith('rec'):
            labels.append(1)
        else:
            print('Wrong type!')
    
    return labels, new_name

train_label,new_name = get_labels(train_dataset)
test_label, _ = get_labels(test_dataset)

# check
print('-'*20,'train_dataset','-'*20)
print(len(train_label))
print(np.unique(train_label))
print('-'*20,'test_dataset','-'*20)
print(len(test_label))
print(np.unique(test_label))


# In[15]:


# train linear SVMs
from sklearn.svm import LinearSVC

svm_hard = LinearSVC(loss='hinge', C=1000).fit(X_train_lsi,train_label)
svm_soft = LinearSVC(loss='hinge', C=0.0001).fit(X_train_lsi,train_label)


# In[20]:


# define plot drawing
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = metrics.auc(fpr,tpr)

    ax.plot(fpr, tpr, linewidth=2,label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)
        
# prepare plot for svm_hard

score_hard = svm_hard.decision_function(X_test_lsi)
#print(len(score_hard))
fpr_hard,tpr_hard,_ = metrics.roc_curve(test_label, score_hard, pos_label=1)

# draw the ROC for svm_hard

plot_roc(fpr_hard, tpr_hard)


# In[19]:


# prepare plot for svm_soft

score_soft = svm_soft.decision_function(X_test_lsi)
fpr_soft, tpr_soft, _ = metrics.roc_curve(test_label, score_soft, pos_label=1)

# draw the ROC for svm_soft

plot_roc(fpr_soft, tpr_soft)


# In[21]:


fig, ax = plt.subplots()

line_hard = ax.plot(fpr_hard, tpr_hard, linewidth=2)
line_soft = ax.plot(fpr_soft, tpr_soft, linewidth=2)

ax.grid(color='0.7', linestyle='--', linewidth=1)

ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate',fontsize=15)
ax.set_ylabel('True Positive Rate',fontsize=15)

ax.legend(('Hard SVM','Soft SVM'),loc="lower right")
plt.title('ROC curve of SVM with C=1000(hard) and C=0.0001(soft)')

for label in ax.get_xticklabels()+ax.get_yticklabels():
    label.set_fontsize(15)


# In[34]:


# Confusion Matrix & other parameters
def evaluate(classifier, clf_name, testSet, label):
    y_pred = classifier.predict(testSet)
    CM = metrics.confusion_matrix(label,y_pred)
    
    print('-'*20, 'Confusion Matrix of',clf_name,'-'*20)
    print(CM)
    
    accuracy = metrics.accuracy_score(label, y_pred)
    recall = metrics.recall_score(label, y_pred)
    precision = metrics.precision_score(label, y_pred)
    Fscore = metrics.f1_score(label, y_pred)
    
    print('-'*20, 'Other Evaluation of',clf_name,'-'*20)
    print('Accuracy:',accuracy)
    print('Recall:',recall)
    print('Precision:',precision)
    print('F-1 Score:',Fscore)

    return CM,accuracy,recall,precision,Fscore

CM_hard,accuracy_hard,recall_hard,precision_hard,Fscore_hard = evaluate(
    svm_hard,'hard SVM(C=1000)',X_test_lsi,test_label)
print()
evaluate(svm_soft,'soft SVM(C=0.0001)',X_test_lsi,test_label)

print()
print('-'*20,'Notes of Index','-'*20)
print('labels are:')
print('Value of 0 ->',new_name[0])
print('Value of 1 ->',new_name[1])


# ### Use cross-validation to choose $\gamma$

# In[35]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

def cross_validate(classifier, train, label):
    avg_score = []

    for para in range(-3,4):
        classifier.set_params(C=10**para).fit(train,label)
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        scores = cross_val_score(classifier,train,label,cv=cv,scoring='accuracy')
    
        print('-'*20,'C =',10**para,'-'*20)
        print(scores)
        print(np.average(scores))
        avg_score.append(np.average(scores))

    max_score = np.max(avg_score)
    index = avg_score.index(max_score)
    c_idx = [x for x in range(-3,4)][index]
    print('-'*20,'Result','-'*20)
    print('The best classifier is when C =',10**c_idx)
    print('Average F-1 Score is:',max_score)
    
    return classifier.set_params(C=10**c_idx).fit(train,label)

svm = cross_validate(LinearSVC(loss='hinge'),X_train_lsi,train_label)


# **$\therefore$ the best $\gamma$ is 10**

# In[31]:


# ROC curve of the best

print(svm)

score = svm.decision_function(X_test_lsi)
fpr, tpr, _ = metrics.roc_curve(test_label, score, pos_label=1)

plot_roc(fpr, tpr)
plt.title('ROC Curve of the best SVM')


# In[32]:


# Confusion Matrix of the best SVM and other parameters
CM,accuracy,recall,precision,Fscore = evaluate(
    svm,'best SVM(C=10)',X_test_lsi,test_label)


# In[33]:


# compare with hard svm 

print('-'*20, 'Other Evaluation of Best SVM(C=10)','-'*20)
print('Accuracy:',accuracy)
print('Recall:',recall)
print('Precision:',precision)
print('F-1 Score:',Fscore)

print('-'*20, 'Other Evaluation of Hard SVM(C=1000)','-'*20)
print('Accuracy:',accuracy_hard)
print('Recall:',recall_hard)
print('Precision:',precision_hard)
print('F-1 Score:',Fscore_hard)


# ## Question 5:
# Logistic classifier:
# - Train a logistic classifier without regularization (you may need to come up with some way to approximate this if you use sklearn.linear_model.LogisticRegression); plot the ROC curve and report the confusion matrix and calculate the **accuracy, recall precision** and **F-1 score** of this classifier.
# - Regularization:
#     * Using 5-fold cross-validation on the dimension-reduced-by-svd training data, find the best regularization strength in the range $\lbrace10^k\mid -3 \leq k \leq 3, k \in \mathbb Z\rbrace$ for logistic regression with L1 regularization and logistic regression L2 regularization, respectively.
#     * Compare the performance (accuracy, precision, recall and F-1 score) of 3 logistic classifiers: w/o regularization, w/L1 regularization and w/L2 regularization (with the best parameters you found from the part above), using test data.
#     * How does the regularization parameter affect the test error? How are the learnt coefficients affected? Why might one be interested in each type of regularization?
#     * Both logistic regression and linear SVM are trying to classify data points using a linear decision boundary, then what's the difference between their ways to find this boundary?
#     <br>Why their performance differ?

# ### Train a logistic classifier without regularization

# In[38]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1',solver='saga',
                          tol=1e-6, max_iter=int(1e6),
                          warm_start=True,random_state=42,C=float(10**6))
clf_nonReg = clf.fit(X_train_lsi, train_label)


# In[39]:


# plot ROC curve
score_nonReg = clf_nonReg.decision_function(X_test_lsi)
fpr_nonReg, tpr_nonReg, _ = metrics.roc_curve(test_label, score_nonReg, pos_label=1)

plot_roc(fpr_nonReg, tpr_nonReg)
plt.title('ROC Curve of Logistic Regression without Regularization')


# In[67]:


# Confusion Matrix of the best SVM
evaluate(clf_nonReg,'Logistic Regression without Regularization (train)',
        X_train_lsi,train_label)
CM_nonReg,accuracy_nonReg,recall_nonReg,precision_nonReg,Fscore_nonReg = evaluate(clf_nonReg,'Logistic Regression without Regularization (test)',
        X_test_lsi,test_label)

print()
print('-'*20,'Notes of Index','-'*20)
print('labels are:')
print('Value of 0 ->',new_name[0])
print('Value of 1 ->',new_name[1])


# ### Logistic Regression with Regularization

# #### Using 5-fold cross-validation to optimize the regularization strength

# In[56]:


# L1 regularization
clf_L1 = cross_validate(LogisticRegression(penalty='l1',solver='saga',
                          tol=1e-6, max_iter=int(1e6),
                          warm_start=True,random_state=42),
                        X_train_lsi,train_label)


# #### $\therefore$ The best inverse of regularization strength for L1 is 1000

# In[57]:


# L2 regularization
clf_L2 = cross_validate(LogisticRegression(penalty='l2',solver='saga',
                          tol=1e-6, max_iter=int(1e6),
                          warm_start=True,random_state=42),
                        X_train_lsi,train_label)


# #### $\therefore$ The best inverse of regularization strength for L2 is 100

# In[43]:


print(clf_nonReg)


# In[44]:


clf_L1


# In[45]:


clf_L2


# #### Compare 3 classifiers

# In[49]:


# For test data, compare their performance

y_pred_L1 = clf_L1.predict(X_test_lsi)
y_pred_L2 = clf_L2.predict(X_test_lsi)

accuracy_L1 = metrics.accuracy_score(test_label, y_pred_L1)
accuracy_L2 = metrics.accuracy_score(test_label, y_pred_L2)

recall_L1 = metrics.recall_score(test_label, y_pred_L1)
recall_L2 = metrics.recall_score(test_label, y_pred_L2)

precision_L1 = metrics.precision_score(test_label, y_pred_L1)
precision_L2 = metrics.precision_score(test_label, y_pred_L2)

Fscore_L1 = metrics.f1_score(test_label, y_pred_L1)
Fscore_L2 = metrics.f1_score(test_label, y_pred_L2)

# show time

print('-'*20, 'Logistic Regression with w/o regularization','-'*20)
print('Accuracy:',accuracy_nonReg)
print('Recall:',recall_nonReg)
print('Precision:',precision_nonReg)
print('F-1 Score:',Fscore_nonReg)

print('-'*20, 'Logistic Regression with w/L1 regularization','-'*20)
print('Accuracy:',accuracy_L1)
print('Recall:',recall_L1)
print('Precision:',precision_L1)
print('F-1 Score:',Fscore_L1)

print('-'*20, 'Logistic Regression with w/L2 regularization','-'*20)
print('Accuracy:',accuracy_L2)
print('Recall:',recall_L2)
print('Precision:',precision_L2)
print('F-1 Score:',Fscore_L2)


# In[68]:


# Confusion Matrix 
CM_L1 = metrics.confusion_matrix(test_label, y_pred_L1)
CM_L2 = metrics.confusion_matrix(test_label, y_pred_L2)

# show time
print('-'*20, 'Confusion Matrix of w/o regularization','-'*20)
print(CM_nonReg)
print('-'*20, 'Confusion Matrix of w/L1 regularization','-'*20)
print(CM_L1)
print('-'*20, 'Confusion Matrix of w/L2 regularization','-'*20)
print(CM_L2)
print('-'*20, 'Notes of Index','-'*20)
print('labels are:')
print('Value of 0 ->',new_name[0])
print('Value of 1 ->',new_name[1])


# In[51]:


# Learnt coefficients
print('-'*20, 'Coefficients of w/o regularization','-'*20)
print(clf_nonReg.coef_)
print('-'*20, 'Coefficients of w/L1 regularization','-'*20)
print(clf_L1.coef_)
print('-'*20, 'Coefficients of w/L2 regularization','-'*20)
print(clf_L2.coef_)


# ## Question 6:
# Naive  Bayes classifier: Train a GaussianNB classifier; plot the ROC curve and report the confusion matrix and calculate the **accuracy, recall, precision** and **F-1 score** of this classifier.

# In[52]:


from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB().fit(X_train_lsi, train_label)


# In[53]:


# ROC curve for test set
score_bayes = bayes.predict_proba(X_test_lsi)[:,1]
fpr_bayes, tpr_bayes, _ = metrics.roc_curve(test_label, score_bayes, pos_label=1)

plot_roc(fpr_bayes, tpr_bayes)
plt.title('ROC Curve of GaussianNB (test set)')


# In[54]:


# ROC curve for train set
score_train_bayes = bayes.predict_proba(X_train_lsi)[:,1]
fpr_train_bayes, tpr_train_bayes, _ = metrics.roc_curve(train_label, score_train_bayes, pos_label=1)

plot_roc(fpr_train_bayes, tpr_train_bayes)
plt.title('ROC Curve of GaussianNB (train set)')


# In[55]:


# Evaluation
evaluate(bayes,'GaussianNB (train set)',X_train_lsi,train_label)

print()

_ = evaluate(bayes,'GaussianNB (test set)',X_test_lsi,test_label)


# ## Question 7:
# Grid search of parameters:
# - Construct a Pipeline that performs feature extraction,, dimensionality reduction and classification;
# - Do grid search with 5-fold cross-validation to compare the following (use test accuracy as the score to compare);
# - What is the best combination?

# #### Pipeline

# In[66]:


# prepare for pipeline
from sklearn.pipeline import Pipeline
np.random.seed(42)
random.seed(42)

pipeline = Pipeline([
    ('vect',CountVectorizer(min_df=1,stop_words='english',analyzer=stem_rmv_punc)),
    ('tfidf',TfidfTransformer()),
    ('reduce_dim',TruncatedSVD(n_components = 50)),
    ('clf',GaussianNB())
])

pipeline.fit(train_dataset.data, train_label)
predict = pipeline.predict(test_dataset.data)
print("accuracy:{}".format(metrics.accuracy_score(test_label, predict)))


# ##### Grid Search
# Compare Table:
# 
# | Procedure | Options |
# | ------- | ----- | 
# | Loading Data | remove "headers" and "footers" vs not | 
# | Feature Extraction | min_df = 3 vs 5 <br> use lemmatization vs not |
# | Dimensionality Reduction | LSI vs NMF |
# | Classifier | SVM with the best $\gamma$ <br> vs <br> Logistic Regression: L1 regularization <br> vs L2 regularization, <br> with the best regularization strength previously found <br> vs <br> GaussianNB |
# | Other options | Use default |

# In[ ]:


# remove headers and footers

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD, NMF

# used to cache results
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
# print(__doc__)
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

categories = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'rec.autos', 'rec.motorcycles',
'rec.sport.baseball', 'rec.sport.hockey']

X_train_noheader = fetch_20newsgroups(subset = 'train', 
                                      categories = categories, 
                                      shuffle = True, 
                                      remove=('headers','footers'),
                                      random_state = None)

label_noheader,new_label = get_labels(X_train_noheader)

pipeline = Pipeline([
    ('vect',CountVectorizer(stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('reduce_dim',TruncatedSVD(n_components = 50)),
    ('clf',GaussianNB())
], memory = memory)

param = [
    {
        'vect':[CountVectorizer(stop_words='english',min_df=3),
                CountVectorizer(stop_words='english',min_df=5),
                CountVectorizer(stop_words='english',min_df=3,analyzer=stem_rmv_punc),
                CountVectorizer(stop_words='english',min_df=5,analyzer=stem_rmv_punc)
               ],
        'reduce_dim': [TruncatedSVD(n_components = 50), NMF(n_components = 50,init='random')],
        'clf': [LinearSVC(loss='hinge',C=10),
                LogisticRegression(penalty='l1',solver='saga',
                          tol=1e-6, max_iter=int(1e6),
                          warm_start=True,C=1000),
                LogisticRegression(penalty='l2',solver='saga',
                          tol=1e-6, max_iter=int(1e6),
                          warm_start=True,C=100),
                GaussianNB()]
    }
]

gs = GridSearchCV(pipeline, param, cv=5, n_jobs=3,
                              verbose=1, scoring='accuracy', 
                              return_train_score=True)
gs.fit(X_train_noheader.data,label_noheader)
rmtree(cachedir)


# In[ ]:


import pandas as pd

pd.DataFrame(grid.cv_results_)


# In[ ]:


# do not remove headers and footers

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

X_train_header = fetch_20newsgroups(subset = 'train', 
                                      categories = categories, 
                                      shuffle = True, 
                                      remove=(),
                                      random_state = None)

label_header,new_label = get_labels(X_train_noheader)
gs_header = GridSearchCV(pipeline,param,cv=5,n_jobs=3,
                        verbose=1,scoring='accuracy',
                        return_train_score=True)
gs_header.fit(X_train_header.data,laber_header)
rmtree(cachedir)


# ## Question 8:
# In this part, we aim to learn classifiers on the documents belonging to the classes:
# 
# <font color=LightCoral>[comp.sys.ibm.pc.hardware, comp.sys.mac.hardware,
# misc.forsale, soc.religion.christian]</font>
# 
# Perform Naive Baayes classification and multiclass SVM classification (with both One VS One and One VS the rest methods described above) and report the **confusion matrix** and calculate the **accuracy, recall, precision** and **F-1 score** of your classifiers.

# In[ ]:


# prepare data
categories8 = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'misc.forsale', 'soc.religion.christian']

X_train8 = fetch_20newsgroups(subset = 'train', 
                                      categories = categories8, 
                                      shuffle = True, 
                                      random_state = None)
X_test8 = fetch_20newsgroups(subset = 'test', 
                                      categories = categories8, 
                                      shuffle = True, 
                                      random_state = None)


# #### Naive Bayes

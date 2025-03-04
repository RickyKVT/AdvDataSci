#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Ricky Truong
# #### Student ID: s3783560
# 
# Date: XXXX
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# This notebook is comprised of 2 tasks. In task 2 we are to generate 3 feature representation for the job description of each file. The first feature is to be a count vector representation feature, where it is based on a bag of word model. The second and third feature representation are a unweighted and weighted vector representation that is to be based on word embeddings using any embedding language model.
# 
# Task 3 conists of 2 different sub-parts. The first sub-parts requires us to build a 2 different classification model whose features are to be based on any of the three feature representation generated in task 2. These models are then to be compared. The second part of task 3 required us to compare if different features of the job text file will provide better results for a classification model. The first model of this task is to be built only on the title of job, the second, just the job description which we have done previously. And the final model is to be based on the title and job description of the job text file. Simiarly, these 3 models will be compared

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import numpy as np
import nltk
import os
import re
from nltk.probability import *
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import gensim.downloader as api


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# 
# 2.1 will build a feature representation based on the bags of words model which 2.2 will build feature represesntation based on word embeddings of a document.

# ### 2.1 Bag of words Language Model
# 
# The bag of words model means that the order of words do not matter. We based our feature based on the number of words or its count.

# Code block below is just to print and confirm if we have successfully built a feature representation.

# In[2]:


#print the feature representation of the first article
def validator(data_features, vocab, tokenised_articles):
        for word, value in zip(vocab, data_features.toarray()[0]):
            if value > 0:
                print(f"{word}:{value}", end=' ')
        print("\n")


# Next we shall read in the description token which we obtained from task 1. Ensure file is in same directory as this notebook

# In[3]:


#read in job description file and tokenise each line
#one line = one job descrption
description_tokens = []
with open('description_tokens.txt', 'r') as f:
    for line in f:
        tokens = list(line.strip().split(',')) #for each line strip the '\n' and split based on ,
        description_tokens.append(tokens) # append tokens list to list of tokens
    f.close()
# print len and first list to confirm
print(len(description_tokens))
print(description_tokens[0])
    


# We also read in the vocab file which we also obtained and saved in task 1

# In[4]:


# read in vocab and save as a dictionary format
#where key:value = word:word index
dict_vocab = {}
with open('vocab.txt', 'r') as f:
    for line in f:
        vocab = line.strip().split(':') # split line based of ':'
        dict_vocab[vocab[0]] = vocab[1] #assign key as first element (word) and value as second element (index)
    f.close()
print(len(dict_vocab))


# The final step is to build the count feature representation

# In[5]:


vocab = list(dict_vocab.keys()) # converts all keys (words) to a list of words = vocab
cVectoriser =CountVectorizer(analyzer= 'word', vocabulary=vocab)

#creates a string of tokens(job description, and for each string fit to count_vectorizer
count_features = cVectoriser.fit_transform(["".join(description) for description in description_tokens])
print(count_features.shape)


# In[6]:


#check if it works by printing first article
validator(count_features, vocab, description_tokens)


# ### 2.2 Word Embeddings Language Model

# ##### Unweighted Vector
# For this section we are using a pre train language model (word2vec), that has been trained on google-news-300. We shall use this model to create word embeddings in our document based on the tokens of descrptions that we have provided.

# The function below generates our web embeddings where it takes our pre-train word embeddings and tokens to produce vector representation based on our tokens. 

# In[7]:


#unweighted word embeddings
#generate vector representation of document
def docvecs(embeddings, tokens):
    vecs = np.zeros((len(tokens), embeddings.vector_size))
    for i, token in enumerate(tokens):
        valid_keys = [term for term in token if term in embeddings.key_to_index] # append words that appear in pretrain and tokens
        if valid_keys: #check for valid key b/c sometimes valid keys is empty and will result in error
            docvec = np.vstack([embeddings[term] for term in valid_keys]) # create vector
            docvec = np.sum(docvec, axis=0) 
            vecs[i,:] = docvec
    return vecs


# In[8]:


#file is 1.6gb so it might take a while to run
preTW2v_wv = api.load('word2vec-google-news-300') # load the pre train model


# Create a dataframe for the tokens as the function to above needs to iterate both the index and tokens of that index

# In[9]:


#creation of dataframe for tokens
df = pd.DataFrame()
df['tokens'] = description_tokens
df['tokens']


# In[10]:


preTW2v_dvs = docvecs(preTW2v_wv,df['tokens']) # generate document vector representation
print('Total number of unweighted vectors: ',preTW2v_dvs.size)


# ##### Weighted Vector
# For this section we will build a weight vector that is TF-IDF weighted word vector. We shall we using the same pre-train model provided in the previous section as well as the list of tokens.

# We will first have to create a mapping between a word in the vocab as its respective weight. To do this we first must get the vocabulary where it maps each word to an index as well as the td-idf vector.
# 

# In[11]:


#vocab that we read in from previous task
dict_vocab


# Next we shall create the TD-IDF vector which acts as the weighted part of this feature representation.

# In[12]:


vocab = list(dict_vocab.keys()) # get list of keys and set as vocab
tVectoriser =TfidfVectorizer(analyzer= 'word', vocabulary=vocab)
tdidf_features = tVectoriser.fit_transform(["".join(description) for description in description_tokens])


# In[13]:


# print tdidf weights for first article 
validator(tdidf_features, vocab, description_tokens)


# We shall write the TD-IDF vector to a file and read it in later as it will easier to work with.

# In[14]:


#saving tdidf vectors to text
def write_tfidfFile(data_features,filename):
    num = data_features.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each article by index
        for f_ind in data_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
            value = data_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
            out_file.write("{}:{} ".format(f_ind,value)) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file
    
write_tfidfFile(tdidf_features,'tVector_file.txt')


# Now we will map each word in the vocab to its respective weight.

# In[15]:


def doc_wordweights(fName_tVectors, voc_dict):
    # a list to store the  word:weight dictionaries of documents, each element if a job description which contains dictionary of 
    # weights for that job.
    tfidf_weights = [] 
    
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector
        tv = tv.strip()
        weights = tv.split(' ') # list of 'word_index:weight' entries
        weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
        key_list = list(voc_dict.keys()) # create list of all keys
        val_list = list(voc_dict.values()) #create all list of values
        wordweight_dict = {} # create dict to store word:weight for each description
        for w in weights:
            position = val_list.index(w[0]) # get the index position of the word
            wordweight_dict[key_list[position]] = w[1] # word at index: weight
        tfidf_weights.append(wordweight_dict) # apppend the dict to list
    return tfidf_weights

tfidf_weights = doc_wordweights('tVector_file.txt', dict_vocab)


# In[16]:


# printing out the weights of the first description
tfidf_weights[0]


# Now we shall generate the weight word embeddings.

# In[17]:


def weighted_docvecs(embeddings, tfidf, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        tf_weights = [float(tfidf[i].get(term, 0.)) for term in valid_keys]
        assert len(valid_keys) == len(tf_weights)
        weighted = [embeddings[term] * w for term, w in zip(valid_keys, tf_weights)]
        if weighted: # same problem as before, some weighted lists are empty, so we only execute if not empty
            docvec = np.vstack(weighted)
            docvec = np.sum(docvec, axis=0)
            vecs[i,:] = docvec
    return vecs

#genearte word embeddings
weighted_preTW2v_dvs = weighted_docvecs(preTW2v_wv, tfidf_weights, df['tokens'])


# In[18]:


weighted_preTW2v_dvs


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[19]:


#read in webindex file and append each line as a element in a list
webindex_list = []
with open('webindex.txt', 'r') as f:
    for line in f:
        line = f.read().split('\n')
        webindex_list = line
print(len(webindex_list))
 


# In[20]:


#function writes count vector to index of word based on vocab text file
def write_vectorFile(data_features,filename,webindex):
    num = data_features.shape[0] # the number of job descriptions
    out_file = open(filename, 'w') 
    for a_ind in range(0, num): # loop through each description
        out_file.write('#'+ webindex[a_ind]) #write out '#' followed by the webindex
        for f_ind in data_features[a_ind].nonzero()[1]: # only get values that are non zero
            value = data_features[a_ind][0,f_ind] # get count value
            
            # write the vector count to the file in the format of word_index:value
            #',' at front to ignore edge cases such as -  if it is the last word count vector
            out_file.write(",{}:{}".format(f_ind,value)) 
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file


# In[21]:


write_vectorFile(count_features,'count_vectors.txt',webindex_list)


# ## Task 3. Job Advertisement Classification

# 3.1 We use the count feature vector representation and unweighted word embeddings to create two classification model in which we shall compare.
# 
# 3.2 We build 2 classification model, one model will just have the features of the job title, the other has both the job title and its description. The third model results are taken from 3.1.

# ### 3.1 Language Model Comparison

# #### 3.1.1 Bag of words Machine Language model

# ##### Functions
# 

# In[22]:


#converts a string to a vector
def str2vec(vec_str,voc_size): # vec_str is a line in the vector txt file, voc_size is the length of the vocab
    doc_vec = [0] * voc_size 
    # processing the vec_str
    vec_str_as_list = vec_str.split(',') # splits string based on ',' NOTE** contains the webindex
    only_vec = vec_str_as_list[1:] # removes the webindex
    for pair in only_vec: # this only contains word_index, freq
        w_ind = int(pair.split(':')[0]) # get the first value which is the word index
        w_freq = float(pair.split(':')[1]) # get the second value which is the freq
        doc_vec[w_ind] = w_freq # dict = {word_index: word_freq}
    return doc_vec


# In[23]:


# takes name of vector representaion and vocab size
# reads each line of the file to construct a maxtrix representation
# converts sparse matrix into CSR matrix
def vecF2matrix(vec_fname,voc_size): 
    with open(vec_fname) as vecf:
        vec_strings = vecf.readlines() # reading a list of strings, each for a document/article
    doc_vectors = [str2vec(vstr.strip(),voc_size) for vstr in vec_strings] # construct the matrix representation for the corpus                                                                  # by calling the 'str2vec' function for each line/string
    return csr_matrix(doc_vectors) # convert the sparse matrix into csr format and return the obtain csr matrix


# In[24]:


#function to evalute mode based on score
def evaluate(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# ##### Reading in vocab and count features

# In[25]:


vocab = 'vocab.txt'
with open(vocab) as f:
    voc_size = len(f.readlines())
print(voc_size)


# In[26]:


count_features = vecF2matrix('count_vectors.txt',voc_size)
print(count_features.shape)


# ##### Reading in Labels and Webindex

# In[27]:


#read all labels in list
labels_list = []
with open('class.txt', 'r') as f:
    labels_list = f.read().split('\n') # read file and split based on newline and assign list as output
print('Output:',labels_list[-1]) #edge case where there is no last element b/c from the way we saved the file
labels_list.pop(-1) # remove last element
print(len(labels_list)) # confirm the it is removed


# In[28]:


print(labels_list[0:10])


# In[29]:


#read all webindex
webindex_list = []
with open('webindex.txt', 'r') as f:
    webindex_list = f.read().split('\n')
print(webindex_list[-1]) # same problem as label edge case of NULL value for last elment
webindex_list.pop(-1)
print(len(webindex_list))


# In[30]:


print(webindex_list[0:10])


# ##### Logistic Regression model based on bags of words language model

# In[31]:


seed = 0


# In[32]:


#splitting the dataset in train and test 
X_train, X_test, y_train, y_test = train_test_split(count_features, labels_list ,test_size=0.2, random_state=seed)


# In[33]:


#using 5 fold validation on test
kf = KFold(n_splits= 5, random_state=seed, shuffle = True) # set up 5 fold validation
fold = 0
score = []
for train_index, test_index in kf.split(list(range(0,len(labels_list)))):
    y_train = [labels_list[i] for i in train_index] # assign y_train index
    y_test = [labels_list[i] for i in test_index] # assign y test index
    X_train_count, X_test_count = count_features[train_index], count_features[test_index] #assign respective xtrain and xtest
    print('Fold',fold+1,': ',evaluate(count_features[train_index],
                                      count_features[test_index],y_train,y_test,seed)) #evaluate each fold
    score.append(evaluate(count_features[train_index],count_features[test_index],y_train,y_test,seed))
    fold +=1
print('Average score:',sum(score)/len(score)) #print average score


# Using the count feature representation we have achieved a average score of 87.11%

# #### 3.1.2 Word Embeddings Language Model Classification
# 

# In[34]:


seed = 0


# In[35]:


#splitting the dataset in train and test 
X_train, X_test, y_train, y_test = train_test_split(tdidf_features, labels_list ,test_size=0.2, random_state=seed)


# In[36]:


#using 5 fold validation on test
kf = KFold(n_splits= 5, random_state=seed, shuffle = True) # set up 5 fold validation
fold = 0
score = []
for train_index, test_index in kf.split(list(range(0,len(labels_list)))):
    y_train = [labels_list[i] for i in train_index] # assign y_train index
    y_test = [labels_list[i] for i in test_index] # assign y test index
    X_train_count, X_test_count = tdidf_features[train_index], tdidf_features[test_index] #assign respective xtrain and xtest
    print('Fold',fold+1,': ',evaluate(tdidf_features[train_index],
                                      tdidf_features[test_index],y_train,y_test,seed)) #evaluate each fold
    score.append(evaluate(tdidf_features[train_index],tdidf_features[test_index],y_train,y_test,seed))
    fold +=1
print('Average score:',sum(score)/len(score)) #print average score


# Using the unweighted word embedding feature representation we have achieved a average score of 88.91%

# In conclusion, the unweighted word embeddings model does a little better than the count feature model which a score of 88.91% compared to 87.11%. This could be due to the word embeddings taking into account the context of words. Altought not tried in this task, the the weighted word embeddings could even prove more accurate since it has the additional parameter of weights which could improve the model. Additionally, different word embeddings model could improve the word embedding model. This this task we utilise word2vec, which is not the most accurate model. Models such as GloVe or FastText could perform better than our model.

# ### 3.2 More information, higher accuracy?

# #### 3.2.1 Just job Title

# In task 1, the only information that we extracted was the title strings, which we saved into a text file. Each line in the title.txt represents the title of one job. In order to build answer this question we have to read in the title.txt and proceed to do all the text pro-processing steps, similar to task 1. Although there will some modifications to the pro-processings steps since we dont have alot of data to work with, compared to the task 1. Next we will create a feature representation for job title, as this will be needed for our model. The final step will be build a model and obtain a accuracy score which will be compared to other models.

# In[37]:


# since title.txt is just list of strings we need to tokenise it
def tokeniseData(description):
    '''
    Function tokenises a description string
    '''
    lower_description = description.lower() # convert all to lowercase
    pattern = r'''(?x)
    [a-zA-Z]+(?:[-'][a-zA-Z]+)? # whole words or words with hyphens/ apostrophe
    '''
    tokenizer = nltk.RegexpTokenizer(pattern) 
    tokenised_description = tokenizer.tokenize(lower_description)
    return tokenised_description


# In[38]:


#read in title.txt
title_list = []
with open('title.txt', 'r') as f:
    title_list = f.read().split('\n')
f.close()

print('last element: ',title_list[-1]) #check last element as some files save nothing as the last element
title_list.pop(-1) # remove last element
print('last element: ',title_list[-1]) # check last element again
print(len(title_list)) # print total titles which should equal to number of jobs which is 776


# In[39]:


#tokenised the list of title strings
tokenised_title = [tokeniseData(title) for title in title_list] 
print(tokenised_title[0]) # check first element of tokenised title to confirm it worked


# Since there a few tokens in each title, we shall only remove the stop words and the top 25 most frequent words based on document frequency, as even with less data we wont be able to obtain a accurate model.
# 
# With few tokens in each title, there is high chance that most words will only appear once in the whole document, therefore if we did remove words that only appear once in document, similar to job descrption, we will remove alot of data, which is higly undesirable.
# 

# In[40]:


# reading in stop words file
stopwords = []
with open('stopwords_en.txt') as f:
    stopwords = f.read().splitlines()
f.close()


# In[41]:


# for each title remove the word if it is in the stopword list
tokenised_title = [[w for w in title if w not in stopwords] for title in tokenised_title]
print(tokenised_title[0])


# Using the same element, we see that we have removed the stop word 'a'

# In[42]:


# removing words that only contain one character
tokenised_title= [[w for w in title if len(w)  >= 2 ] for title in tokenised_title] 
print(tokenised_title[0])


# In[43]:


#get the 3 most common words based on document frequency
words = list(chain.from_iterable([set(title)                                     for title in tokenised_title])) # get set of unique words for that article
doc_freq = FreqDist(words)
most_freq_doc = []
freq_doc = doc_freq.most_common(25) #output : list of tuple (word,freq)
#append 50 most common words to a list
for i in freq_doc:
    most_freq_doc.append(i[0])
print(most_freq_doc)#check the list if they contain the words


# We can see that the word 'title', 'manager' and 'sales' are top 3 most common words based on document frequency.

# In[44]:


#for each title remove the 25 most common words based on document frequency if they exist in the title
tokenised_title = [[w for w in title if w not in most_freq_doc] for title in tokenised_title]
print(tokenised_title[0])


# From the first title tokens we see that the token title has been removed.
# 
# In order to compare without any bias, the features generated will have to be similar, therefore we will build a count vector for the title tokens similar to what we did with the job description. 

# In[45]:


#generate vocab based on title tokens
words_final = list(chain.from_iterable(tokenised_title)) #flatten the tokenised words of descriptions
vocab_title = sorted(set(words_final))
print(vocab_title[0:10]) # sample of vocab


# In[46]:


#generate count features for title
cVectoriser =CountVectorizer(analyzer= 'word', vocabulary=vocab_title)

#creates a string of tokens(job description, and for each string fit to count_vectorizer
title_count_features = cVectoriser.fit_transform([" ".join(title) for title in tokenised_title])
print(title_count_features.shape) # (number of job title, size of vocab)


# In[47]:


#validate the count features buy printing first count for first job title
validator(title_count_features, vocab_title, tokenised_title)


# In[48]:


# create x and y train and test
X_train, X_test, y_train, y_test = train_test_split(title_count_features, labels_list ,test_size=0.2, random_state=seed)


# In[49]:


#using 5 fold validation on test
kf = KFold(n_splits= 5, random_state=seed, shuffle = True) # set up 5 fold validation
fold = 0
score = []
for train_index, test_index in kf.split(list(range(0,len(labels_list)))):
    y_train = [labels_list[i] for i in train_index] # assign y_train index
    y_test = [labels_list[i] for i in test_index] # assign y test index
    X_train_count, X_test_count = title_count_features[train_index], title_count_features[test_index] #assign respective xtrain and xtest
    print('Fold',fold+1,': ',evaluate(title_count_features[train_index],
                                      title_count_features[test_index],y_train,y_test,seed)) #evaluate each fold
    score.append(evaluate(title_count_features[train_index],title_count_features[test_index],y_train,y_test,seed))
    fold +=1
print('Average score:',sum(score)/len(score)) #print average score


# Average accuracy score for the classification model on just using the title as input data is 69.2%

# #### 3.2.2 Just job Description

# Text pro-processing was done in task 1. Feature generations on the pro-processed text was obtained during task 2. The classifcation modelling on just the description data was done during the first half of task 3. The results are repeated below for clarity.

# In[50]:


#using 5 fold validation on test
kf = KFold(n_splits= 5, random_state=seed, shuffle = True) # set up 5 fold validation
fold = 0
score = []
for train_index, test_index in kf.split(list(range(0,len(labels_list)))):
    y_train = [labels_list[i] for i in train_index] # assign y_train index
    y_test = [labels_list[i] for i in test_index] # assign y test index
    X_train_count, X_test_count = count_features[train_index], count_features[test_index] #assign respective xtrain and xtest
    print('Fold',fold+1,': ',evaluate(count_features[train_index],
                                      count_features[test_index],y_train,y_test,seed)) #evaluate each fold
    score.append(evaluate(count_features[train_index],count_features[test_index],y_train,y_test,seed))
    fold +=1
print('Average score:',sum(score)/len(score)) #print average score


# Average accuracy score for the classification model on just using the title as input data is 87.1%.

# #### 3.2.3 Both Title and Job Description

# To build a model on both the texts of title and job description we will concatenate the tokens of the title with the tokens of the job description. They way we have saved or extract tokens means the first element in the title tokens correlates to the first element in the description tokens. No text pro-processing is required since we already have done it previously, but we have to build a new vocabulary. The next step would be to generate a count vector feature for consistency, where we will finally use the feature to build a classification model and gague the accuracy of the model.

# We shall first print the first element of each list to see the structure of the list. We shall also print the length of each list to ensure they are both the same.

# In[51]:


#printing 
print(tokenised_title[0])
print(len(tokenised_title))


# In[52]:


print(description_tokens[0])
print(len(description_tokens))


# In[53]:


#concat the the tokens elementwise via zip function
t_and_d = []
for title, description in zip(tokenised_title, description_tokens):
    concat = title + description
    t_and_d.append(concat)
print('Total: ',len(t_and_d))    


# In[54]:


#print first element
print(t_and_d[0])


# Looking at the first concatenated title and description tokens the format for the tokens are in consistent with some words containing white space while others do not. We shall fix this to keep it consistent.
# 

# In[55]:


t_and_d = [[w.strip() for w in tokens] for tokens in t_and_d] # strip whitespace
print(t_and_d[0]) 


# We have strip all whitespace in each token, to ensure the same format for each token. Now we can start pre-processing the tokens an build a meaningful vocabulary.

# In[56]:


#generate vocab 
words_final = list(chain.from_iterable(t_and_d)) #flatten the tokenised words of descriptions
vocab_t_and_d= sorted(set(words_final))
print(vocab_t_and_d[0:10]) # sample of vocab
print(len(vocab_t_and_d))


# Notice how this vocab is much larger than the two other vocab we have built on just title and just description

# In[57]:


cVectoriser =CountVectorizer(analyzer= 'word', vocabulary=vocab_t_and_d)

#creates a string of tokens(job description, and for each string fit to count_vectorizer
t_and_d_count_features = cVectoriser.fit_transform([" ".join(token) for token in t_and_d])
print(t_and_d_count_features.shape) # (number of job title, size of vocab)


# In[58]:


# validate the count feature generation
validator(t_and_d_count_features, vocab_t_and_d, t_and_d)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(t_and_d_count_features, labels_list ,test_size=0.2, random_state=seed)


# In[60]:


#using 5 fold validation on test
kf = KFold(n_splits= 5, random_state=seed, shuffle = True) # set up 5 fold validation
fold = 0
score = []
for train_index, test_index in kf.split(list(range(0,len(labels_list)))):
    y_train = [labels_list[i] for i in train_index] # assign y_train index
    y_test = [labels_list[i] for i in test_index] # assign y test index
    X_train_count, X_test_count = t_and_d_count_features[train_index], title_count_features[test_index] #assign respective xtrain and xtest
    print('Fold',fold+1,': ',evaluate(t_and_d_count_features[train_index],
                                      t_and_d_count_features[test_index],y_train,y_test,seed)) #evaluate each fold
    score.append(evaluate(t_and_d_count_features[train_index],t_and_d_count_features[test_index],y_train,y_test,seed))
    fold +=1
print('Average score:',sum(score)/len(score)) #print average score


# Average accuracy score for the classification model on just using the title as input data is 88.6%.

# In conlusions, the results above suggests that the more information provided, to create a feature representation, will improve the model when classify which class the job will belong to. Further can be done, where we could even include company as well as title and job description. We could also use different feature representation, such as word embeddings as this one utilises count vectors.

# ## Summary
# Give a short summary and anything you would like to talk about the assessment tasks here.
# In summary, the word embeddings features provided a more accurate model than the bag of word model(count vector), which could be due that it takes in the context of the words. Further testing on different word embeddings language model to generate the features could provide a more accurate model. Additionally, the more information that is provided in order to generate a feature representation, suggests that it will always result in a more accurate model, which makes sense as the more data that is availiable, the more accurate the model will be.
# 
# ***NOTE ENSURE THAT ALL REQUIRED TEXT FILES ARE IN THE SAME DIRECTORY AS THIS NOTEBOOK.

# In[ ]:





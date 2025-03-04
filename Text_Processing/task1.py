#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
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
# * re
# * numpy
# * itertools
# * os
# * nltk
# 
# ## Introduction
# This this notebook we are provided with hundreds of text files each corresponding to a different classes, that are 'Accounting Finance', 'Engineering', 'Healthcare Nursing' or 'Sales'. Our goal is to apply basic text pro-processings steps to an job description that is extract from each file. Each file contains the following: a title, a web index, a job description and some may also have a company name.
# 

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
from itertools import chain
import numpy as np
import nltk
import os
import re
from nltk.probability import *


# ### 1.1 Examining and loading data
# - Examine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# We will first inspect the file directory

# In[2]:


# Code to inspect the provided data file...
files = os.listdir("data") #get list of files in the "data" folder
print(files)


# In[3]:


files = os.listdir("data/Accounting_Finance") # get list of all files in 'Accounting_Finance' folder
print(len(files))


# In[4]:


files = os.listdir("data/Engineering/") # get list of all files in Engineering folder
print(len(files))


# In[5]:


files = os.listdir("data/Healthcare_Nursing/") # get list of all files in Healthcare_Nursing folder
print(len(files))


# In[6]:


files = os.listdir("data/Sales/")# get list of all files in Sales folder
print(len(files))


# In[7]:


#read the first file in the sales folder
with open("data/Sales/Job_00621.txt", 'r') as f:
    data = f.readlines()
    f.close()
data


# In[8]:


#read the first file in the engineering folder
with open("data/Engineering/Job_00017.txt", 'r') as f:
    data = f.readlines()
    f.close()
data


# In[9]:


#read the first file in the engineering folder
with open("data/Healthcare_Nursing/Job_00426.txt", 'r') as f:
    data = f.readlines()
    f.close()
data[-1][13:]


# From the above code blocks, we can see that the data folder contains 4 folders not including the hidden folder. The folder names are 'Accounting_Finance', 'Engineering', 'Healthcare_Nursing', and 'Sales'. Each folder contains 191, 231, 198,  and 156 text files, respectively, for a total of 776 text files. Reading the text files, we can see that each text file contains the title of the job, a webindex, a description of the job and some of them, not all will also contain the company for which the job is for.

# Next we shall extract job description and some additional information from the text files as they will come in handy in the later tasks

# In[10]:


dir_path = "./data"
job_description = []
job_index = []
target_class = []
title = []
for folder in os.listdir(dir_path):# load all folders
    if folder != '.DS_Store': #anything folder but the hidden folder '.DS_Store'
        path = os.path.join(dir_path,folder) # e.g /data/Engineering
        for filename in os.listdir(path): #load all text files
            new_path = os.path.join(path,filename) # e.g /data/Engineering/Job_00017.txt
            with open(new_path,"r",encoding = 'unicode_escape') as f:
                data = (f.readlines())
                title.append(data[0].strip()) #append title of each job to list
                target_class.append(str(folder))#save the folder name as the class
                job_index.append(data[1][10:]) #get the webindex from the digits onwards
                
                job_description.append(data[-1]) #data[-1] obtains the last lines which is the description blocks and appends to list
                f.close()
print(len(job_description)) # length of list should equal total number of files which is 776
        


# In[11]:


#randomly pick a job description and webindex to see if its right
print(job_index[485])
print(job_description[485])


# We have successfully extracted the job description of all files and can now proceed with text pre-processing.

# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.

# Following the assignment guidelines we will process the descriptions of the jobs in the following way.
# - Tokenisation of each job description string
# - Convert all words to lower case
# - Remove all words with less than 2 characters
# - Remove all stopwords, with the given stopwords file
# - Remove words that appear only once, based on term frequency
# - Remove the top 50 most common wors based on document frequency
# 

# #### 1.2.1 Tokenisation

# The following function tokenise strings and converts all characters to lowercase.

# In[12]:


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


# In[13]:


# list comprehension to tokenise every description in list
tokenised_description = [tokeniseData(description) for description in job_description] 


# In[14]:


#print to check results of random descprition 
print(tokenised_description[485])
print(len(tokenised_description))


# We can confirm that all strings are tokenised.

# #### 1.2.2 Removing words

# The code block below removes all tokens with only one character.

# In[15]:


#for each description in the list only keep words longer or equal to 2 characters
tokenised_description = [[w for w in description if len(w)  >= 2 ] for description in tokenised_description] 
print(tokenised_description[485])
print(len(tokenised_description[485]))


# The following removes stopwords from the list of tokens based on a stopwords file.

# In[16]:


#read stop words file and add to a list
stopwords = []
with open('stopwords_en.txt') as f:
    stopwords = f.read().splitlines()
    f.close()
print(stopwords[0:10]) # print first 10 stop words


# In[17]:


# for each description remove the word if it is in the stopword list
tokenised_description = [[w for w in description if w not in stopwords] for description in tokenised_description]
print(len(tokenised_description[485]))


# The section below removes words that only appear once based on term frequency.

# In[18]:


words = list(chain.from_iterable(tokenised_description)) #flatten the list of tokenised description
term_freq = FreqDist(words) #get frequencies of words
less_freq = set(term_freq.hapaxes()) #get the set of words that appear only once.
print(len(less_freq))


# In[19]:


#for each description remove words that appear only once in document collection
tokenised_description = [[w for w in description if w not in less_freq] for description in tokenised_description]
print(len(tokenised_description[485]))


# The section below removes the top 50 most frequent words based on document frequency.

# In[20]:


#get the 50 most frequent words based on document frequency
words_2 = list(chain.from_iterable([set(description)                                     for description in tokenised_description])) # get set of unique words for that article
doc_freq = FreqDist(words_2)
most_freq_doc = []
freq_doc = doc_freq.most_common(50) #output : list of tuple (word,freq)
#append 50 most common words to a list
for i in freq_doc:
    most_freq_doc.append(i[0])
print(most_freq_doc)#check the list if they contain the words


# In[21]:


#for each description remove top 50 most common words based on document freq
tokenised_description = [[w for w in description if w not in most_freq_doc] for description in tokenised_description]
print(len(tokenised_description[485]))


# In[22]:


print(tokenised_description[485])


# ## Saving required outputs
# Now that we have finish the pre-processing steps we shall now save the vocab as specified in the assignment specification as it will be required in task 2 and 3.
# 
# Additionally, we shall also save:
# - list of tokens for each job description
# - web index
# - class of each job 
# - title of each job
# 
# as these will be also be used in task 2 and 3, but are not required in the submission

# In[23]:


# code to save pre-processed data for task 2 and 3
def save_tokens(filename,description_tokens):
    token_file = open(filename, 'w')
    string = '\n'. join([', '.join(token) for token in description_tokens]) #for each description append tokens with ','before /n at the end
    token_file.write(string)
    token_file.close()
def save_webindex(filename,webindex):
    webindex_file = open(filename, 'w')
    #save the webindex on each line
    for i in webindex:
        string = str(i)
        webindex_file.write(string)
    webindex_file.close
def save_class(filename,target_class):
    class_file = open(filename,'w')
    #save the target class on each line
    for i in target_class:
        string = str(i) + '\n'
        class_file.write(string)
    class_file.close()
def save_title(filename, title):
    title_file = open(filename,'w',encoding="utf-8") # need encoding for char such as ****
    for i in title:
        string = str(i) + '\n' 
        title_file.write(string) # write each title to a new line
    title_file.close()
        
    


# In[24]:


save_tokens('description_tokens.txt',tokenised_description)
save_webindex('webindex.txt',job_index)
save_class('class.txt',target_class)
save_title('title.txt',title)

print('Number of instances (tokens): ', len(tokenised_description))
print('Number of instances: (job index): ', len(job_index))
print('Number of instances: (class): ', len(target_class))
print('Number of instances (title): ',len(title))


# In[25]:


#save vocab to vocab.txt
words_final = list(chain.from_iterable(tokenised_description)) #flatten the tokenised words of descriptions
vocab = sorted(set(words_final)) # get all unique and save as vocab in alphabetical order
print(len(vocab))
vocab_file = open('vocab.txt','w')
for ind in range(len(vocab)): # for each word in vocab list print index s
     vocab_file.write("{}:{}\n".format(vocab[ind],ind))
vocab_file.close()


# ## Summary
# In summary, in this task we have we extract the description portion of each file, were it underwent basic text pre-processing. This gave us the vocabulary for this task which we saved. Additionally, we also saved the tokens of the pre-processing steps as well as web index, title and class of each text file. As all of these saved output files will be used in both task 2 and 3.
# 
# ***NOTE ENSURE THAT ALL OUTPUTS ARE SAVED INTO THE SAME DIRECTORY AS IT WILL BE CALLED IN TASK 2 AND 3 AND WILL ONLY WORK IF THEY ARE INS THE SAME DIRECTORY.

# In[ ]:






# In[1]:


import pyspark
import re
import numpy as np


# In[2]:


# pyspark works best with java8 
# set JAVA_HOME enviroment variable to java8 path 
#get_ipython().magic('env JAVA_HOME = /usr/lib/jvm/java-8-openjdk-amd64')


# In[3]:


sc = pyspark.SparkContext()


# ## Task 1 - compute "bag of words" for each document (25 pts)
# 
# For task 1, we want to extract "bag of words" features for documents. 
# 
# The first part of this task is the same as what you've already implemented in `Lab - Spark introduction (Vocareum)`. We need a dictionary, as an RDD, that includes the 20,000 most frequent words
# in the training corpus. The result of such an RDD must be in this format:
# `
# [('mostcommonword', 0),
#  ('nextmostcommonword', 1),
#  ...]
# `
# 
# **NOTE**: There aren’t 20,000 unique words in the small dataset (`20_news_same_line_random_sample.txt`). Use only the top 50 words when working with this file.
# 
# For this part, we provided our code, so that you only need to run it, to create this dictionary, named `refDict`, as an RDD. This `refDict` RDD will be our reference dictionary of words. **The words in `refDict` will be our reference words for which we will compute "bag of words" and "TF-IDF" features for our training corpus and finally for the test documents.**

# **Provided code to create the reference dictionary of words.**
# 
# Run the code cells below to create the `refDict` RDD.

# In[4]:


# set the number of dictionary words 
# 50 for the small dataset
# 20,000 for the large dataset
numWords = 20000


# In[5]:


# load up the dataset 
# "data/20_news_same_line_random_sample.txt" for small dataset 
# "s3://comp643bucket/lab/spark_intro_aws/20_news_same_line.txt" for entire large dataset
corpus = sc.textFile ("s3://comp643bucket/lab/spark_intro_aws/20_news_same_line.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id=' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('"> ') + 3:x.index(' </doc>')]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents                       
regex = re.compile('[^a-zA-Z]')  
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top numWords (50 for small dataset, 20K for large dataset) frequent words in a local array
topWords = allCounts.top (numWords, lambda x : x[1])

# and we'll create an RDD that has a bunch of (word, rank) pairs
# start by creating an RDD that has the number 0 up to numWords (50 for small dataset, 20K for large dataset) 
# numWords is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(numWords))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 0) ("nextmostcommon", 1), ...
# the number will be the spot in the dictionary used to tell us where the word is located
refDict = twentyK.map(lambda x:(topWords[x][0],x))



# start your code here ...
# keyAndListOfWords.take(1)

# make values of keyAndListOfWords into key to enable join to refDict
values_to_keys = keyAndListOfWords.flatMap(lambda x: ((word, x[0]) for word in x[1]))
# values_to_keys.take(10)

#join the switched key/values to refDict and select relevant data only
joined = values_to_keys.join(refDict).map(lambda x: x[1])
# joined.take(10)

# Group the data by keys and map the values
grouped_data = joined.groupByKey().mapValues(list)
# grouped_data.take(1)


# In[11]:


# Create a function to count the occurrences of each index in the NumPy arrays
def count_indices(arr):
    counts = np.zeros(numWords)
    unique, index_counts = np.unique(arr, return_counts=True)
    counts[unique] = index_counts
    return counts

# Map each key to a NumPy array representing the count of each index in the original NumPy array
bag_of_words = grouped_data.mapValues(lambda arr: count_indices(arr))
# bag_of_words.take(2)


# Once you created your `bag_of_words` RDD, print out the result arrays for these documents,
# * `20_newsgroups/soc.religion.christian/21626`
# * `20_newsgroups/talk.politics.misc/179019`
# * `20_newsgroups/rec.autos/103167`
# 
# by running the code cells below:

# In[12]:


arr1_1 = np.array(bag_of_words.filter(lambda x: x[0]=='20_newsgroups/soc.religion.christian/21626').values().collect())
# arr1_1[arr1_1.nonzero()]


# In[13]:


arr1_2 = np.array(bag_of_words.filter(lambda x: x[0]=='20_newsgroups/talk.politics.misc/179019').values().collect())
# arr1_2[arr1_2.nonzero()]


# In[14]:


arr1_3 = np.array(bag_of_words.filter(lambda x: x[0]=='20_newsgroups/rec.autos/103167').values().collect())
# arr1_3[arr1_3.nonzero()]

print('TASK 1:')
# arr1_1 = np.array(bag_of_words.filter(lambda x: x[0]=='20_newsgroups/soc.religion.christian/21626').values().collect())
print("20_newsgroups/soc.religion.christian/21626 ... \n", arr1_1[arr1_1.nonzero()])

# arr1_2 = np.array(bag_of_words.filter(lambda x: x[0]=='20_newsgroups/talk.politics.misc/179019').values().collect())
print("20_newsgroups/talk.politics.misc/179019 ...\n", arr1_2[arr1_2.nonzero()])

# arr1_3 = np.array(bag_of_words.filter(lambda x: x[0]=='20_newsgroups/rec.autos/103167').values().collect())
print("20_newsgroups/rec.autos/103167 ... \n", arr1_3[arr1_3.nonzero()])
# ```


# In[1]:


import pyspark
import re
import numpy as np
from collections import Counter


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




# ## Task 2 - compute TF-IDF for each document (30 pts)
# 
# It is often difficult to classify documents accurately using raw count vectors (bag of words). Thus, the next task is
# to write some more Spark code that converts each of the count vectors to TF-IDF vectors. You need to create an RDD of key-value pairs, named `tfidf`, that the keys are document identifiers, and the values are the TF-IDF vector. Again, we are only interested in the top `numWords` (50 for small dataset, 20K for large dataset) most common words as our features.  
# 
# The item index i in a TF-IDF vector of document d corresponds to the TF-IDF value of the word with rank i in `refDict`. Then, TF-IDF value of a word with rank i for document $d$ is computed as:
# 
# $$ TF(i, d) \times IDF(i) $$
# 
# Where $TF(i, d)$ is: 
# 
# $$ \frac {\textrm{Number of occurances of word with rank $i$ in $d$}} {\textrm{Total number of refDict words in $d$}} $$
# 
# Note that the “Total number of `refDict` words” is not the number of distinct words. The “total number of words”
# in “Today is a great day today” is six. 
# 
# And the $IDF(i)$ is:
# 
# $$ \ln \frac {\textrm{Number of documents in corpus}} {\textrm{Number of documents having word with rank $i$}} $$
# 
# Once you created this `tfidf` RDD, print out the non-zero array entries (TF-IDF vector) that you have created for these documents:
# 
# * 20_newsgroups/soc.religion.christian/21626
# * 20_newsgroups/talk.politics.misc/179019
# * 20_newsgroups/rec.autos/103167
# 
# <font color=red> Important: If you are using `bag_of_words` RDD, don't collect it into a Python object; work with it as an RDD </font>  

# In[15]:


# start your code here ...

# Total number of refDict words in d
refdict_wrds_in_d = bag_of_words.map(lambda x: (x[0], sum(x[1])))
# refdict_wrds_in_d.take(1)

#calculate the tf
tf = bag_of_words.join(refdict_wrds_in_d).map(lambda doc: (doc[0], doc[1][0]/doc[1][1]))
# tf.take(1)


# In[16]:


import math 
# number of doc in the corpus
docs_in_corpus = keyAndListOfWords.count()
# docs_in_corpus

#get distict docs and words
distinct_doc_w_word = refDict.join(values_to_keys).map(lambda x: (x[0], x[1][1])).distinct() 

# count number of docs with each word.
doc_w_word_cnt = distinct_doc_w_word.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
doc_w_word_cnt = doc_w_word_cnt.map(lambda x: (x[0], math.log(docs_in_corpus/x[1])))
# doc_w_word_cnt.take(10)


# In[17]:


idf = doc_w_word_cnt.map(lambda x: ('word', x[1])).groupByKey().mapValues(list)
idf = np.array(idf.collect()[0][1])
# idf


# In[18]:


tfidf = tf.map(lambda x: (x[0], x[1]*idf))
# tfidf.take(1)



# ## Task 3 - build a kNN classifier (30 pts)
# 
# Task 3 is to build a kNN classifier, as a Python function named `predictLabel` in the cell below. This function will take as input a text string (`test_doc`) and a number k, and then output the name of one of the 20 newsgroups. This name is the news group that the classifier thinks that the text string is “closest” to. It is computed using the classical kNN algorithm. 
# 
# Your function first needs to convert the input string into all lower case words, and then compute a TF-IDF vector corresponding to the words in `refDict` created in the first task. Recall that the words in `refDict` is our reference words to compute "TF-IDF" features. In task 2, we already computed TF-IDF values of these words for our training corpus. In this task, you need to compute TF-IDF values of these words for the input text string `test_doc`. For that, you need to compute term frequency of these words in the `test_doc`. Since IDF measure of a word only depends on the training corpus, and this measure is already calculated for `refDict` words in task 2, you don't need to re-calculate IDF for the `test_doc` and can re-use what you have.    
# Then, your function needs to find the k documents in the corpus that are “closest” to the `test_doc` (where distance is computed using the l2 norm between the TF-IDF feature vectors), and returns the newsgroup label that is most frequent in those top k. Ties go to the label with the closest corpus document. 
# 
# Once you have implemented your function, run it on the following 8 test cases, each is an excerpt from a Wikipedia article,
# chosen to match one of the 20 newsgroups. By reading each test document, you can guess which of the 20 newsgroups is the most relevent topic, and you can compare that with what your prediction function returns. The result you get from the small dataset might not be so accurate, due to the small training corpus. But, once you run it on the entire dataset in S3, you should see reasonable results (with few mis-matches).  
# 
# <font color=red>Important: `refDict` is an RDD and must stay as an RDD as you work with; don't collect it into a Python object to work with</font> 

# In[22]:


# k, test_doc = (10, 'Graphics are pictures and movies created using computers – usually referring to image data created by a computer specifically with help from specialized graphical hardware and software. It is a vast and recent area in computer science. The phrase was coined by computer graphics researchers Verne Hudson and William Fetter of Boeing in 1960. It is often abbreviated as CG, though sometimes erroneously referred to as CGI. Important topics in computer graphics include user interface design, sprite graphics, vector graphics, 3D modeling, shaders, GPU design, implicit surface visualization with ray tracing, and computer vision, among others. The overall methodology depends heavily on the underlying sciences of geometry, optics, and physics. Computer graphics is responsible for displaying art and image data effectively and meaningfully to the user, and processing image data received from the physical world. The interaction and understanding of computers and interpretation of data has been made easier because of computer graphics. Computer graphic development has had a significant impact on many types of media and has revolutionized animation, movies, advertising, video games, and graphic design generally.')
# test_doc = sc.parallelize(test_doc.lower().split())
# test_doc_dict = test_doc.map(lambda x: (x, 1)).reduceByKey (lambda a, b: a + b)
# # test_doc_dict.take(10)
# # refDict.take(10)

# test_doc_joined = refDict.join(test_doc_dict).map(lambda x: (x[1][0], x[1][1]))

# # test_doc_joined.collect()


# In[23]:


# test_doc_counts = np.zeros(numWords)

# for x in test_doc_joined.collect():
#     test_doc_counts[x[0]] = x[1] 
# # test_doc_counts

# test_doc_tf = test_doc_counts/sum(test_doc_counts)
# test_doc_tfidf = test_doc_tf * idf
# # test_doc_tfidf

# # distance = np.linalg.norm(a - b)


# In[24]:


# # calculate the distance
# distance = tfidf.map(lambda x: (x[0], np.linalg.norm(test_doc_tfidf - x[1])))
# # distance.take(3)

# # select the closest
# closest = distance.sortBy(lambda x: x[1]).take(k)
# output = [x[0] for x in closest]
# # output


# In[25]:


# k is the number of neighbors to consider
# test_doc is the text to compare 
def predictLabel (k, test_doc):
    # your code here
    test_doc = sc.parallelize(test_doc.lower().split())
    test_doc_dict = test_doc.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
    # test_doc_dict.take(10)
#     refDict.take(10)

    test_doc_joined = refDict.join(test_doc_dict).map(lambda x: (x[1][0], x[1][1]))
    
    test_doc_counts = np.zeros(numWords)

    for x in test_doc_joined.collect():
        test_doc_counts[x[0]] = x[1] 
    # test_doc_counts

    test_doc_tf = test_doc_counts/sum(test_doc_counts)
    test_doc_tfidf = test_doc_tf * idf
#     test_doc_tfidf
    # calculate the distance
    distance = tfidf.map(lambda x: (x[0], np.linalg.norm(test_doc_tfidf - x[1])))
    closest = distance.sortBy(lambda x: x[1]).take(k)
    #output = [x[0] for x in closest]
    output = [x[0].split('/')[1] for x in closest]
    counter = Counter(output)
    return counter.most_common(1)[0][0]


print('TASK 3:')
print(predictLabel (10, 'Graphics are pictures and movies created using computers – usually referring to image data created by a computer specifically with help from specialized graphical hardware and software. It is a vast and recent area in computer science. The phrase was coined by computer graphics researchers Verne Hudson and William Fetter of Boeing in 1960. It is often abbreviated as CG, though sometimes erroneously referred to as CGI. Important topics in computer graphics include user interface design, sprite graphics, vector graphics, 3D modeling, shaders, GPU design, implicit surface visualization with ray tracing, and computer vision, among others. The overall methodology depends heavily on the underlying sciences of geometry, optics, and physics. Computer graphics is responsible for displaying art and image data effectively and meaningfully to the user, and processing image data received from the physical world. The interaction and understanding of computers and interpretation of data has been made easier because of computer graphics. Computer graphic development has had a significant impact on many types of media and has revolutionized animation, movies, advertising, video games, and graphic design generally.'))
print(predictLabel (10, 'A deity is a concept conceived in diverse ways in various cultures, typically as a natural or supernatural being considered divine or sacred. Monotheistic religions accept only one Deity (predominantly referred to as God), polytheistic religions accept and worship multiple deities, henotheistic religions accept one supreme deity without denying other deities considering them as equivalent aspects of the same divine principle, while several non-theistic religions deny any supreme eternal creator deity but accept a pantheon of deities which live, die and are reborn just like any other being. A male deity is a god, while a female deity is a goddess. The Oxford reference defines deity as a god or goddess (in a polytheistic religion), or anything revered as divine. C. Scott Littleton defines a deity as a being with powers greater than those of ordinary humans, but who interacts with humans, positively or negatively, in ways that carry humans to new levels of consciousness beyond the grounded preoccupations of ordinary life.'))
print(predictLabel (10, 'Egypt, officially the Arab Republic of Egypt, is a transcontinental country spanning the northeast corner of Africa and southwest corner of Asia by a land bridge formed by the Sinai Peninsula. Egypt is a Mediterranean country bordered by the Gaza Strip and Israel to the northeast, the Gulf of Aqaba to the east, the Red Sea to the east and south, Sudan to the south, and Libya to the west. Across the Gulf of Aqaba lies Jordan, and across from the Sinai Peninsula lies Saudi Arabia, although Jordan and Saudi Arabia do not share a land border with Egypt. It is the worlds only contiguous Eurafrasian nation. Egypt has among the longest histories of any modern country, emerging as one of the worlds first nation states in the tenth millennium BC. Considered a cradle of civilisation, Ancient Egypt experienced some of the earliest developments of writing, agriculture, urbanisation, organised religion and central government. Iconic monuments such as the Giza Necropolis and its Great Sphinx, as well the ruins of Memphis, Thebes, Karnak, and the Valley of the Kings, reflect this legacy and remain a significant focus of archaeological study and popular interest worldwide. Egypts rich cultural heritage is an integral part of its national identity, which has endured, and at times assimilated, various foreign influences, including Greek, Persian, Roman, Arab, Ottoman, and European. One of the earliest centers of Christianity, Egypt was Islamised in the seventh century and remains a predominantly Muslim country, albeit with a significant Christian minority.'))
print(predictLabel (10, 'The term atheism originated from the Greek atheos, meaning without god(s), used as a pejorative term applied to those thought to reject the gods worshiped by the larger society. With the spread of freethought, skeptical inquiry, and subsequent increase in criticism of religion, application of the term narrowed in scope. The first individuals to identify themselves using the word atheist lived in the 18th century during the Age of Enlightenment. The French Revolution, noted for its unprecedented atheism, witnessed the first major political movement in history to advocate for the supremacy of human reason. Arguments for atheism range from the philosophical to social and historical approaches. Rationales for not believing in deities include arguments that there is a lack of empirical evidence; the problem of evil; the argument from inconsistent revelations; the rejection of concepts that cannot be falsified; and the argument from nonbelief. Although some atheists have adopted secular philosophies (eg. humanism and skepticism), there is no one ideology or set of behaviors to which all atheists adhere.'))

'''
print(predictLabel (10, 'President Dwight D. Eisenhower established NASA in 1958 with a distinctly civilian (rather than military) orientation encouraging peaceful applications in space science. The National Aeronautics and Space Act was passed on July 29, 1958, disestablishing NASAs predecessor, the National Advisory Committee for Aeronautics (NACA). The new agency became operational on October 1, 1958. Since that time, most US space exploration efforts have been led by NASA, including the Apollo moon-landing missions, the Skylab space station, and later the Space Shuttle. Currently, NASA is supporting the International Space Station and is overseeing the development of the Orion Multi-Purpose Crew Vehicle, the Space Launch System and Commercial Crew vehicles. The agency is also responsible for the Launch Services Program (LSP) which provides oversight of launch operations and countdown management for unmanned NASA launches.'))
print(predictLabel (10, 'The transistor is the fundamental building block of modern electronic devices, and is ubiquitous in modern electronic systems. First conceived by Julius Lilienfeld in 1926 and practically implemented in 1947 by American physicists John Bardeen, Walter Brattain, and William Shockley, the transistor revolutionized the field of electronics, and paved the way for smaller and cheaper radios, calculators, and computers, among other things. The transistor is on the list of IEEE milestones in electronics, and Bardeen, Brattain, and Shockley shared the 1956 Nobel Prize in Physics for their achievement.'))
print(predictLabel (10, 'The Colt Single Action Army which is also known as the Single Action Army, SAA, Model P, Peacemaker, M1873, and Colt .45 is a single-action revolver with a revolving cylinder holding six metallic cartridges. It was designed for the U.S. government service revolver trials of 1872 by Colts Patent Firearms Manufacturing Company – todays Colts Manufacturing Company – and was adopted as the standard military service revolver until 1892. The Colt SAA has been offered in over 30 different calibers and various barrel lengths. Its overall appearance has remained consistent since 1873. Colt has discontinued its production twice, but brought it back due to popular demand. The revolver was popular with ranchers, lawmen, and outlaws alike, but as of the early 21st century, models are mostly bought by collectors and re-enactors. Its design has influenced the production of numerous other models from other companies.'))
print(predictLabel (10, 'Howe was recruited by the Red Wings and made his NHL debut in 1946. He led the league in scoring each year from 1950 to 1954, then again in 1957 and 1963. He ranked among the top ten in league scoring for 21 consecutive years and set a league record for points in a season (95) in 1953. He won the Stanley Cup with the Red Wings four times, won six Hart Trophies as the leagues most valuable player, and won six Art Ross Trophies as the leading scorer. Howe retired in 1971 and was inducted into the Hockey Hall of Fame the next year. However, he came back two years later to join his sons Mark and Marty on the Houston Aeros of the WHA. Although in his mid-40s, he scored over 100 points twice in six years. He made a brief return to the NHL in 1979–80, playing one season with the Hartford Whalers, then retired at the age of 52. His involvement with the WHA was central to their brief pre-NHL merger success and forced the NHL to expand their recruitment to European talent and to expand to new markets.'))
'''
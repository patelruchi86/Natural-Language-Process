#!/usr/bin/env python
# coding: utf-8

# 
# ## News Analysis using Topic Modeling & NLTK

# ### 1. Install the the necessary packages and libraries

# In[1]:


import requests
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
from pprint import pprint
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import pyLDAvis.gensim_models
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


# ### 2. Load the data from google news API

# In[2]:


pip install newsapi-python


# In[2]:


from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='**********')


# In[4]:


sources = newsapi.get_sources(country='us')
list = []
sources = sources['sources'] # getting the list of news portals with google
for i in sources:
  list.append(i['id']) # saving the id of each portals

top_headlines = []
for i in list:
  top_headline = newsapi.get_top_headlines(sources=i,) # getting top headlines of each portal
  for j in top_headline['articles']:
    top_headlines.append(j)
print(top_headlines)

titles = []
for i in top_headlines:      
  titles.append(i['title'])  # getting titles of each news

df = pd.DataFrame(titles)
df.columns = ['titles'] # saving all titles in dataframe
print(df)


# ### 3. Clean the Data

# In[20]:


news_df = df
tokenized_doc = news_df['titles'].str.replace("[^a-zA-Z#]", " ")
news_df.shape


# In[21]:


def clean_doc(text):
    #for token in text:
    text =  ' '.join([w.lower() for w in text.split() if len(w)>3])
    return text;

tokenized_doc = tokenized_doc.apply(clean_doc)
tokenized_doc



# In[22]:


stopwords_set = stopwords.words("english")

def text_preproc(x):
  x = x.lower()
  x = ' '.join([word for word in x.split(' ') if word not in stop_words]) # remove stopwords
  x = x.encode('ascii', 'ignore').decode() # Remove unicode characters
  x = re.sub(r'https*\S+', ' ', x) # Remove URL
  x = re.sub(r'@\S+', ' ', x) # Remove mentions
  x = re.sub(r'#\S+', ' ', x) # Remove Hashtags
  x = re.sub(r'\'\w+', '', x) # Remove ticks and the next character
  x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x) # Remove punctuations
  # x = re.sub(r'\w*\d+\w*', '', x) # Remove numbers
  x = re.sub(r'\s{2,}', ' ', x) # Replace the over spaces
  return x

tokenized_doc = tokenized_doc.apply(text_preproc)
tokenized_doc


# ### 4. Featurization using TFIDF

# In[14]:


# Initialising the tfidf vectorizer with the default stopword list 
tfidf = TfidfVectorizer(stop_words="english", max_features= 1000, max_df = 0.5, smooth_idf=True)

#Vectorizing 'X' column
vector =tfidf.fit_transform(tokenized_doc)

#Converting vector into an array
X= vector.toarray()
pd.DataFrame(X)


# In[15]:


stop = stopwords_set
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Function to lemmatize and remove the stopwords
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Creating a list of documents from the complaints column
list_of_docs = tokenized_doc.tolist()

# Implementing the function for all the complaints of list_of_docs
doc_clean = [clean(doc).split() for doc in list_of_docs]
print(doc_clean[:2])


# ### 5. Perform the topic modelling using genism's LDA

# In[16]:


# Creating the dictionary id2word from our cleaned word list doc_clean
dictionary = corpora.Dictionary(doc_clean)

# Creating the corpus
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the LDA model
ldamodel = LdaModel(corpus=doc_term_matrix, num_topics=5,id2word=dictionary, random_state=20, passes=30)

# printing the topics
pprint(ldamodel.print_topics())


# #### Here we can see the top keywords and their weights associated with that topic. words with high probability in topic and with associate probabilities in topic distribution. 
# #### By looking at keywords we can guess that topic 3 is related to space, topic 0 related to politics etc.

# ### 6. Evaluate the model with preplexity & coherence score

# In[17]:


# Compute Perplexity
perplexity_lda = ldamodel.log_perplexity(doc_term_matrix)
print('\nPerplexity: ', perplexity_lda)  


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# #### Lower the perplexity indicates the better model.
# #### Higher the topic coherence score, the topic is more interpretable for human.

# ### 7. Visulize the result using pyLDAvis and wordclouds

# In[18]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(ldamodel, doc_term_matrix, dictionary)
vis



# #### Each cluster on the left side associate with some category. The size of the bubble indicate the importance of the topic in the entire documents.
# #### A good model is where fairly big topics covered in different quadrants rather than one quadrants.
# #### here we can see the topic 4 & 5 overlapping meaning they are closely related to each other. Probably we can say it is business and political group.
# #### If we hover over the cursor on the  different bubbles we can see different keywords associated with that topics.
# 
# #### to find the optimal no of cluster, one can build LDA model with diffirent values of the cluster and check for the better evaluation score. Here I have choose the 5 topics for LDA model.
# 
# #### Finally, we need to understand that this is an unsupervised techniques, and we need to judge the topic to assign a particular category and it is challanging part.
# 
# #### From the occurrence of the word, we can categorized Topic 1 as  world political issues, Topic 2 as E-commerce businesses, Topic 3 as space technology, Topic 4 & 5 closely relatives to political and business category.
# 

# In[19]:


cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stopwords,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = ldamodel.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
fig.savefig('word_cloud.png')
plt.show()


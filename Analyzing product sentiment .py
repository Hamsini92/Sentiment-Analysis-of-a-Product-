
# coding: utf-8

# In[10]:

import graphlab 


# # read some product review data 

# In[11]:

products = graphlab.SFrame('amazon_baby.gl')


# In[12]:

products 


# In[13]:

products.head()


# In[14]:

products.tail()


# # build the word count vector for each review 

# In[16]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[17]:

products.head()


# In[18]:

graphlab.canvas.set_target('ipynb')


# In[19]:

products['name'].show()


# # explore vulli sophie 

# In[20]:

giraffee_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[22]:

len(giraffee_reviews)


# In[23]:

giraffee_reviews['rating'].show(view='Categorical')


# # Build a sentimental classifier 

# In[25]:

products['rating'].show(view='Categorical')


# # define whats a positive and a negative sentiment 

# In[26]:

#ignore all 3 * reviews 
products = products[products['rating'] != 3]


# In[27]:

#positive sentiment = 4* or 5* reviews 
products['sentiment'] = products['rating'] >= 4 


# In[28]:

products.head()


# # lets train the sentiment classifier 

# In[29]:

train_data, test_data = products.random_split(.8, seed=0)


# In[30]:

sentiment_model = graphlab.logistic_classifier.create(train_data, target='sentiment', features=['word_count'], validation_set=test_data)


# # evaluate the sentiment model 

# In[31]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[32]:

sentiment_model.show(view='Evaluation')


# # applying the learned model to understand the sentiment for girraffee reviews 

# In[36]:

giraffee_reviews['predicted_sentiment'] = sentiment_model.predict(giraffee_reviews, output_type='probability')


# In[37]:

giraffee_reviews.head()


# # sort the reviews based on the predicted sentiment and explore 

# In[38]:

giraffee_reviews = giraffee_reviews.sort('predicted_sentiment', ascending=False)


# In[39]:

giraffee_reviews.head()


# In[40]:

giraffee_reviews[0]['review']


# In[41]:

giraffee_reviews[1]['review']


# # show most negative reviews 

# In[42]:

giraffee_reviews[-1]['review']


# In[43]:

giraffee_reviews[-2]['review']


# In[ ]:




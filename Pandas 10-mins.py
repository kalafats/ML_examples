#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


s = pd.Series([1, 3, 5, np.nan, 6, 8])


# In[7]:


s


# In[8]:


type(s)


# In[9]:


dates = pd.date_range('20130101', periods=6)


# In[10]:


dates


# In[11]:


df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))


# In[12]:


df


# Creating a DataFrame by passing a dict of objects that can be converted to series-like.

# In[13]:


df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})


# In[14]:


df2


# The columns of the resulting DataFrame have different dtypes.

# In[15]:


df2.dtypes


# Here is how to view the top and bottom rows of the frame:

# In[16]:


df.head()


# In[17]:


df.tail(3)


# Display the index, columns:

# In[18]:


df.index


# In[19]:


df.columns


# DataFrame.to_numpy() gives a NumPy representation of the underlying data. Note that this can be an expensive operation when your DataFrame has columns with different data types, which comes down to a fundamental difference between pandas and NumPy: NumPy arrays have one dtype for the entire array, while pandas DataFrames have one dtype per column. When you call DataFrame.to_numpy(), pandas will find the NumPy dtype that can hold all of the dtypes in the DataFrame. This may end up being object, which requires casting every value to a Python object.
# 
# For df, our DataFrame of all floating-point values, DataFrame.to_numpy() is fast and doesn’t require copying data.

# In[20]:


df.to_numpy()


# For df2, the DataFrame with multiple dtypes, DataFrame.to_numpy() is relatively expensive.

# In[21]:


df2.to_numpy()


# describe() shows a quick statistic summary of your data:

# In[22]:


df.describe()


# Transposing your data:

# In[23]:


df.T


# Sorting by an axis:

# In[24]:


df.sort_index(axis=1, ascending=False)


# Sorting by values:

# In[25]:


df.sort_values(by='B')


# ### Selection

# Selecting a single column, which yields a Series, equivalent to df.A:

# In[26]:


df['A']


# Selecting via [], which slices the rows.

# In[27]:


df[0:3]


# In[28]:


df['20130102':'20130104']


# In[29]:


df


# In[30]:


df.loc[dates[0]]


# In[31]:


type(df.loc[dates[0]])


# Selecting on a multi-axis by label:

# In[32]:


df.loc[:, ['A', 'B']]


# Showing label slicing, both endpoints are included:

# In[33]:


df.loc['20130102':'20130104', ['A', 'B']]


# Reduction in the dimensions of the returned object:

# In[35]:


type(df.loc['20130102':'20130104', ['A', 'B']])


# In[34]:


df.loc['20130102', ['A', 'B']]


# In[36]:


type(df.loc['20130102', ['A', 'B']])


# For getting a scalar value:

# In[37]:


df.loc[dates[0], 'A']


# For getting fast access to a scalar (equivalent to the prior method):

# In[38]:


df.at[dates[0], 'A']


# Select via the position of the passed integers: (by index no)

# In[39]:


df.iloc[3]


# In[46]:


df.iloc[3,2]


# In[42]:


df.loc[dates[3]]


# By integer slices, acting similar to numpy/python:

# In[43]:


df.iloc[3:5, 0:2]


# By lists of integer position locations, similar to the numpy/python style:

# In[47]:


df.iloc[[1, 2, 4], [0, 2]]


# For slicing rows explicitly:

# In[49]:


df.iloc[1:3, :]


# In[50]:


df.iloc[1:3]


# For slicing columns explicitly:

# In[51]:


df.iloc[:, 1:3]


# For getting a value explicitly:

# In[53]:


df.iloc[1, 1]


# For getting fast access to a scalar (equivalent to the prior method):

# In[54]:


df.iat[1, 1]


# ### Boolean indexing

# In[55]:


df


# Using a single column’s values to select data.

# In[56]:


df[df.A > 0]


# Selecting values from a DataFrame where a boolean condition is met.

# In[57]:


df[df>0]


# Using the isin() method for filtering:

# In[58]:


df2 = df.copy()


# In[59]:


df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']


# In[60]:


df2


# In[61]:


df2[df2['E'].isin(['two','four'])]


# ### Setting

# Setting a new column automatically aligns the data by the indexes.

# In[63]:


s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))


# In[64]:


s1


# In[65]:


df['F'] = s1


# In[66]:


df


# Setting values by label:

# In[67]:


df.at[dates[0], 'A'] = 0


# Setting values by position:

# In[68]:


df.iat[0, 1] = 0


# Setting by assigning with a NumPy array:

# In[70]:


df.loc[:, 'D'] = np.array([5] * len(df))


# In[71]:


df


# A where operation with setting.

# In[72]:


df2 = df.copy()


# In[73]:


df2[df2 > 0] = -df2


# In[74]:


df2


# ### Missing Data

# pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations.

# Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.

# In[75]:


df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])


# In[76]:


df1.loc[dates[0]:dates[1], 'E'] = 1


# In[77]:


df1


# To drop any rows that have missing data.

# In[78]:


df1.dropna(how='any')


# Filling missing data.

# In[79]:


df1.fillna(value=5)


# To get the boolean mask where values are nan.

# In[80]:


pd.isna(df1)


# ### Operations

# Operations in general exclude missing data.
# 
# Performing a descriptive statistic:

# In[82]:


df.mean()


# Same operation on the other axis:

# In[83]:


df.mean(1)


# Operating with objects that have different dimensionality and need alignment. In addition, pandas automatically broadcasts along the specified dimension.

# In[85]:


s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)


# In[89]:


s


# In[88]:


df.sub(s, axis='index') # subtract


# ### Apply

# In[90]:


df


# Applying functions to the data:

# In[91]:


df.apply(np.cumsum)


# In[92]:


df.apply(lambda x: x.max() - x.min())


# ### String Methods

# In[94]:


s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])


# In[95]:


s


# In[96]:


s.str.lower()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[291]:


import tweepy
import csv
import pandas as pd
import time
import sys
import datetime


# In[269]:


#Twitter API credentials
consumer_key = "BmvoUUcOUXhPxRR8uRC2TgKoW"
consumer_secret = "bNV6inRgeUSSVerytnnnTPveW8iM9GM0dwryZyiUKmYy436D1I"
access_key = "2969993776-b9Ui7fVJjW7gYId2C0kSGo5mN4ki93HSGEn6jx0"
access_secret = "N5ER33zjeIqfl5918MWTHLWbZzuBGfGL0FeSfNGvSsrvZ"


# In[270]:


OAUTH_KEYS = {'consumer_key':consumer_key, 'consumer_secret':consumer_secret, 'access_token_key':access_key, 'access_token_secret':access_secret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[329]:


#Define number of tweets scraped for the time window 
#(None: all, in theory, but when I tried, the limit is still 10000 ...)
n_items_default = None


# ### Functions

# In[330]:


def is_retweeted(text):
    return text[:2] == 'RT'


# In[331]:


def user_retweeted(tweet):
    try:
        RT_info = tweet.split(':')[0]
        if '@' in RT_info:
            user_RT = RT_info.split()[-1]
            user_RT = user_RT.replace('@','')
            return user_RT
        else:
            return False
        pass
    except IndexError as ve:
        print(tweet)
        return False


# In[332]:


def scraping(since_date, until_date, n_items=10000):
    if n_items is None:
        print('No limit')
        cursor = tweepy.Cursor(api.search, q='#confinement',
                               geocode="48.85717,2.34293,10km",
                               since=since_date,
                               until=until_date).items()
    else:
        print('Limit: ', n_items)
        cursor = tweepy.Cursor(api.search, q='#confinement',
                               geocode="48.85717,2.34293,10km",
                               since=since_date,
                               until=until_date).items(n_items)
    tweet_list = []
    while True:
        try:
            tweet = cursor.next()
            tweet_list.append(tweet)
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
        except StopIteration:
            break
    print('Number of tweets scrapped: ', len(tweet_list))

    return tweet_list


# In[333]:


def scrap_to_df(tweet_list):
    
    usernames = []
    text = []
    timestamp = []
    count_rt = []
    for tweet in tweet_list:
        timestamp.append(tweet.created_at)
        usernames.append(tweet.user.screen_name)
        text.append(tweet.text)
        count_rt.append(tweet.retweet_count)
        
    df = pd.DataFrame()
    df['timestamp'] = pd.to_datetime(timestamp)
    max_date = max(df.timestamp)
    min_date = min(df.timestamp)
    df['username'] = usernames
    df['count_rt'] = count_rt
    df['text'] = text
    df['is_retweeted'] = df.text.apply(is_retweeted)
    df['user_retweeted'] = df.text.apply(user_retweeted)
    print('Finished from ', min_date, ' to ', max_date)
    return df


# ### Running

# In[334]:


#dates = ['2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31',
#        '2020-04-01', '2020-04-02', '2020-04-03']
dates = ['2020-04-03', '2020-04-04', '2020-04-05', '2020-04-06']

# #### With only 10 000 items per time window

# In[335]:


def main(dates=dates, n_items=n_items_default):
    print('Dates that will generates windows ')
    print(dates)
    print(n_items_default)
    for i in range(len(dates) - 1):
        print()
        t1 = time.time()
        tweet_list = scraping(dates[i], dates[i+1], n_items=n_items_default)
        df = scrap_to_df(tweet_list)
        csv_name = 'data_'+ str(dates[i]) + '_' + str(dates[i+1]) + '.csv'
        t2 = time.time()
        delta_t = t2-t1
        print('Time taken (sec):', delta_t)
        print('Name of saved csv file: ', csv_name)
        print('Shape of the dataframe: ', df.shape)
        df.to_csv(csv_name,index = False, encoding='utf-8')
        print()
    return 'finished'


# In[336]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





import tweepy as tw
import streamlit as st
import pandas as pd
from transformers import pipeline
from PIL import Image


st.set_page_config(layout="centered")
image = Image.open('/Users/aameerkhan/Desktop/Streamlit-multipage/assests/Tweet by Aameer Khan.png')

st.image(image, caption='Tweet by the creator',width=500)



st.title('Live News about the financial data from Twitter ')
st.markdown('This page helps you to search about the financial and stock market data tweets that are made on the twitter application.')
st.markdown(' This app can be helpful knowing about the current trend.')

st.text("")

st.markdown("Enter you access credentials only then you will be able to search for the tweets.")
consumer_key= st.text_input(label="Enter the consumer key")
consumer_secret=st.text_input(label="Enter the consumer secret key")
access_token=st.text_input(label="Enter the access token")
access_token_secret=st.text_input(label="Enter the access secret token")
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

st.text("")
classifier = pipeline('sentiment-analysis')

def run():
    with st.form(key="Enter name"):
        st.text("")
        search_words = st.text_input('Enter the word that you need to search')
        number_of_tweets = st.number_input('Enter the number of latest tweets that you want to generate (Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            tweets =tw.Cursor(api.search_tweets,q=search_words,lang="en").items(number_of_tweets)
            tweet_list = [i.text for i in tweets]
            p = [i for i in classifier(tweet_list)]
            q=[p[i]['label'] for i in range(len(p))]
            df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Latest' +str(number_of_tweets)+' Tweets'+' on '+search_words, 'sentiment'])
            st.write(df)
        else:  
             st.markdown("If you are not able to generate tweets kindly look into your access credentials and if you have twitter developer account.")


if __name__=='__main__':
    
    run()
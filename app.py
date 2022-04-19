# from statistics import mode
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests
import pandas as pd
import numpy as np
from newspaper import Article
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from textblob import TextBlob
import datefinder
import datetime  
from datetime import date
from PIL import Image, ImageEnhance
import pickle
import streamlit as st
import nltk
nltk.download('stopwords', 'word_tokenize', 'words')
# import streamlit_authenticator as stauth
# from bs4 import BeautifulSoup


# Loading Random Forest model
# model1 = pickle.load(open('Random_Forest', 'rb'))

# Loading RidgeCV model
model2 = pickle.load(open('RidgeCV', 'rb'))

stopwords=set(stopwords.words('english'))

def rate_unique(words):
    words=tokenize(words)
    no_order = list(set(words))
    rate_unique=len(no_order)/len(words)
    return rate_unique


def rate_nonstop(words):
    words=tokenize(words)
    filtered_sentence = [w for w in words if not w in stopwords]
    rate_nonstop=len(filtered_sentence)/len(words)
    no_order = list(set(filtered_sentence))
    rate_unique_nonstop=len(no_order)/len(words)
    return rate_nonstop,rate_unique_nonstop

def avg_token(words):
    words=tokenize(words)
    length=[]
    for i in words:
        length.append(len(i))
    return np.average(length)


def day(article_text):
    article=article_text
    if len(list(datefinder.find_dates(article)))>0:
        date=str(list(datefinder.find_dates(article))[0])
        date=date.split()
        date=date[0]
        year, month, day = date.split('-')     
        day_name = datetime.date(int(year), int(month), int(day)) 
        return day_name.strftime("%A")
    return "Monday"


def tokenize(text):
    text=text
    return word_tokenize(text)


def polar(words):
    pos_words=[]
    neg_words=[]
    all_tokens=tokenize(words)
    for i in all_tokens:
        analysis=TextBlob(i)
        polarity=analysis.sentiment.polarity
        if polarity>0:
            pos_words.append(i)
        if polarity<0:
            neg_words.append(i)
    return pos_words,neg_words


def rates(words):
    words=polar(words)
    pos=words[0]
    neg=words[1]
    all_words=words
    global_rate_positive_words=(len(pos)/len(all_words))/100
    global_rate_negative_words=(len(neg)/len(all_words))/100
    pol_pos=[]
    pol_neg=[]
    for i in pos:
        analysis=TextBlob(i)
        pol_pos.append(analysis.sentiment.polarity)
        avg_positive_polarity=analysis.sentiment.polarity
        
    if len(pol_pos) == 0:
        pol_pos.append(0)

    for j in neg:
        analysis2=TextBlob(j)
        pol_neg.append(analysis2.sentiment.polarity)
        avg_negative_polarity=analysis2.sentiment.polarity

    if len(pol_neg) == 0:
        pol_neg.append(0)

    min_positive_polarity=min(pol_pos)
    max_positive_polarity=max(pol_pos)
    avg_positive_polarity=np.average(pol_pos)
    min_negative_polarity=min(pol_neg)
    max_negative_polarity=max(pol_neg)
    avg_negative_polarity=np.average(pol_neg)
    return global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,min_positive_polarity,max_positive_polarity,avg_negative_polarity,min_negative_polarity,max_negative_polarity


def create_df(url):
    df2=[]
    i = url
    pred_info={}
    article = Article(i, language="en") # en for English 
    article.download() 
    article.parse()
    analysis=TextBlob(article.text)
    polarity=analysis.sentiment.polarity
    title_analysis=TextBlob(article.title)
    pred_info['text']=article.text
    pred_info['n_tokens_title']=len(tokenize(article.title))
    pred_info['n_tokens_content']=len(tokenize(article.text))
    pred_info['n_unique_tokens']=rate_unique(article.text)
    pred_info['n_non_stop_words']=rate_nonstop(article.text)[0]
    pred_info['n_non_stop_unique_tokens']=rate_nonstop(article.text)[1]
    pred_info['num_hrefs']=article.html.count("https://timesofindia.indiatimes.com")
    pred_info['num_imgs']=len(article.images)
    pred_info['num_videos']=len(article.movies)
    pred_info['average_token_length']=avg_token(article.text)
    pred_info['num_keywords']=len(article.keywords)
    
    if "life-style" in article.url:
        pred_info['data_channel_is_lifestyle']=1
    else:
        pred_info['data_channel_is_lifestyle']=0
    if "etimes" in article.url:
        pred_info['data_channel_is_entertainment']=1
    else:
        pred_info['data_channel_is_entertainment']=0
    if "business" in article.url:
        pred_info['data_channel_is_bus']=1
    else:
        pred_info['data_channel_is_bus']=0
    if "social media" or "facebook" or "whatsapp" in article.text.lower():
        data_channel_is_socmed=1
        data_channel_is_tech=0
        data_channel_is_world=0
    else:
        data_channel_is_socmed=0
    if ("technology" or "tech" in article.text.lower()) or ("technology" or "tech" in article.url):
        data_channel_is_tech=1
        data_channel_is_socmed=0
        data_channel_is_world=0
    else:
        data_channel_is_tech=0
    if "world" in article.url:
        data_channel_is_world=1
        data_channel_is_tech=0
        data_channel_is_socmed=0
    else:
        data_channel_is_world=0
        
    pred_info['data_channel_is_socmed']=data_channel_is_socmed
    pred_info['data_channel_is_tech']=data_channel_is_tech
    pred_info['data_channel_is_world']=data_channel_is_world
    
    if day(i)=="Monday":
        pred_info['weekday_is_monday']=1
    else:
        pred_info['weekday_is_monday']=0
    if day(i)=="Tuesday":
        pred_info['weekday_is_tuesday']=1
    else:
        pred_info['weekday_is_tuesday']=0
    if day(i)=="Wednesday":
        pred_info['weekday_is_wednesday']=1
    else:
        pred_info['weekday_is_wednesday']=0
    if day(i)=="Thursday":
        pred_info['weekday_is_thursday']=1
    else:
        pred_info['weekday_is_thursday']=0
    if day(i)=="Friday":
        pred_info['weekday_is_friday']=1
    else:
        pred_info['weekday_is_friday']=0
    if day(i)=="Saturday":
        pred_info['weekday_is_saturday']=1
        pred_info['is_weekend']=1
    else:
        pred_info['weekday_is_saturday']=0
    if day(i)=="Sunday":
        pred_info['weekday_is_sunday']=1
        pred_info['is_weekend']=1
    else:
        pred_info['weekday_is_sunday']=0
        pred_info['is_weekend']=0
        
    pred_info['global_subjectivity']=analysis.sentiment.subjectivity
    pred_info['global_sentiment_polarity']=analysis.sentiment.polarity
    pred_info['global_rate_positive_words']=rates(article.text)[0]
    pred_info['global_rate_negative_words']=rates(article.text)[1]
    pred_info['avg_positive_polarity']=rates(article.text)[2]
    pred_info['min_positive_polarity']=rates(article.text)[3]
    pred_info['max_positive_polarity']=rates(article.text)[4]
    pred_info['avg_negative_polarity']=rates(article.text)[5]
    pred_info['min_negative_polarity']=rates(article.text)[6]
    pred_info['max_negative_polarity']=rates(article.text)[7]    
    pred_info['title_subjectivity']=title_analysis.sentiment.subjectivity
    pred_info['title_sentiment_polarity']=title_analysis.sentiment.polarity
    df2.append(pred_info)
    
    pred_df=pd.DataFrame(df2)
    pred_df=pred_df.drop(['text'],axis=1)
    return pred_df
 
# Create test dataframe from provided article url
# dataf = create_df("https://indianexpress.com/article/explained/what-downgrade-in-average-monsoon-rainfall-means-7869816/")
# print(dataf.head(5))


# predicting the shares
# model.predict(dataf)

st.title("News Article Popularity Prediction") 

heading = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
          ' font-size: 20px;">Developed by Team Ethereal</p'
st.write(heading, unsafe_allow_html=True)

image_file_path = "img.jpeg"

add_dropbox = st.sidebar.selectbox(
    "Choose Input Source",
    ("Home", "User Check")
)

if add_dropbox == "Home":
    image = np.array(Image.open(image_file_path))
    st.image(image)
    st.write("An machine learning approach to predict the number of shares that an online news article could get.")


elif add_dropbox == "User Check":
    image = np.array(Image.open(image_file_path))
    st.image(image)

    input1 = st.text_input("Enter the URL of News Article: ")

    if st.button("Check"):
            dataf = create_df(input1)
#             result = model1.predict(dataf)
            result = model2.predict(dataf)

            # msg = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
            #     ' font-size: 20px;">Predicted virality is:</p'
            msg = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
                ' font-size: 20px;">Predicted Number of Shares:</p'
            st.write(msg, unsafe_allow_html=True)
            st.write(result[0])

    # Filters = st.radio("Select any prediction model:",
    #                         ("Random Forest Model", "Regression Model")
    #                         )

    # input1 = st.text_input("Enter the URL of News Article: ")

    # if Filters == "Random Forest Model":
        
    #     if st.button("Check"):
    #         dataf = create_df(input1)
    #         result = model1.predict(dataf)
            
    #         # msg = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
    #         #     ' font-size: 20px;">Predicted virality is:</p'
    #         msg = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
    #             ' font-size: 20px;">Predicted Number of Shares:</p'
    #         st.write(msg, unsafe_allow_html=True)
    #         st.write(result[0])

    # elif Filters == "Regression Model":
    
    #     if st.button("Check"):
    #         dataf = create_df(input1)
    #         result = model2.predict(dataf)
            
    #         msg = '<p style = "font-family: Franklin Gothic; color: #F63366;' \
    #             ' font-size: 20px;">Predicted Number of Shares:</p'
    #         st.write(msg, unsafe_allow_html=True)
    #         st.write(result[0])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:58:30 2023

@author: kieranmartin
"""
import pandas as pd
from textblob import TextBlob
import os
import spacy
from data_wrangling import get_clean_articles_df
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from wordcloud import WordCloud
from wordcloud import STOPWORDS

# Assign political leaning to News Outlets 
# source: https://libguides.com.edu/c.php?g=649909&p=4556556
def map_political_leaning(outlet):
    if outlet == 'MSNBC':
        return 'Far Left'
    elif outlet in ['CNN', 'NBC News', 'Politico', 'The Washington Post', \
                    'USA Today', 'CBS News', 'Associated Press', 'Vice News']:
        return 'Left'
    elif outlet in ['BBC News', 'Reuters', 'The Hill', 'ABC News']:
        return 'Centrist'
    elif outlet == 'The Wall Street Journal':
        return 'Right'
    elif outlet == 'Fox News':
        return 'Far Right'
    else:
        return 'Unknown'  


def filter_articles():
    articles_df = get_clean_articles_df()

    # First keep all titles that mention keywords that appear by themselves to 
    # filter out 'Israel-Hamas War' to avoid irrelevant subjective coding.
    pattern_keep = r'(Israel*.\s|\sHamas|Gaza|Palestin)'
    mask_keep = articles_df['Title'].str.contains(pattern_keep, case=False, na=False)
    articles_df = articles_df[mask_keep]

    # Titles with keywords still may be off topic and need to be filtered out 
    phrases_to_exclude = ['List of key events', 'Adesanya', '2023', 'Andorra', \
                          'Euro 2024', 'GOP', 'Left', 'Democrat', 'Republican']
    pattern_remove = '|'.join(phrases_to_exclude)
    mask_remove = ~articles_df['Title'].str.contains(pattern_remove, na=False, regex=True)
    articles_df = articles_df[mask_remove]

    # Strip titles of certain unneeded phrases to imporve NLP analysis
    phrases_to_strip = ['- report', '- opinion', '- editorial', '- comment', \
                        '| CNN', '| 60 Minutes', 'WATCH:']
    
    for phrase in phrases_to_strip:
        articles_df['Title'] = articles_df['Title'].str.replace(phrase, '', regex=False)

    political_leaning = articles_df['News Outlet'].apply(map_political_leaning)
    news_outlet_position = articles_df.columns.get_loc('News Outlet')
    articles_df.insert(news_outlet_position + 1, 'Political Leaning', political_leaning)
    
    return articles_df


# Filter out any non-US outlets for analysis
def filter_news_outlets(df):
    values_to_remove = ['Al Jazeera English', 'The Jerusalem Post']
    
    return df[~df['News Outlet'].isin(values_to_remove)]


nlp = spacy.load("en_core_web_sm")


def get_subjectivity(text):
    
    return TextBlob(text).sentiment.subjectivity


def get_intensity(text):
    analyzer = SentimentIntensityAnalyzer()
    
    return analyzer.polarity_scores(text)['compound']


def analyze_article_sentiments(articles_df):
    articles_df['subjectivity'] = articles_df['Title'].apply(get_subjectivity)
    articles_df['intensity'] = articles_df['Title'].apply(get_intensity)

    return articles_df


def process_articles_for_tfidf(df, max_features=2000):
    df['Date Published'] = pd.to_datetime(df['Date Published']).dt.date
    grouped_titles = df.groupby('Date Published')['Title'].apply(lambda x: ' '.join(x))
    counts = CountVectorizer(strip_accents='unicode', lowercase=True, \
                             stop_words='english', max_features=max_features)
    x_counts = counts.fit_transform(grouped_titles)
    tf_trans = TfidfTransformer(use_idf=False)
    x_tfidf = tf_trans.fit_transform(x_counts)
    cols = counts.get_feature_names_out()
    df_tfidf_time = pd.DataFrame(x_tfidf.todense(), columns=cols, \
                                 index=grouped_titles.index)

    return df_tfidf_time


# Split titles into seperate dataframes for comparitive analysis
def get_israel_only_df(articles_df):
    israel_df = articles_df[articles_df['Title'].str.contains(r'Israel.*|IDF*.', \
                            case=False, regex=True)]
    israel_only_df = israel_df[~israel_df['Title'].str.contains(r'Hamas.*', \
                            case=False, regex=True)]
    
    return israel_only_df


def get_hamas_only_df(articles_df):
    hamas_df = articles_df[articles_df['Title'].str.contains(r'Hamas.*', \
                           case=False, regex=True)]
    hamas_only_df = hamas_df[~hamas_df['Title'].str.contains(r'Israel.*|IDF*.',\
                          case=False, regex=True)]
    
    return hamas_only_df


def get_gaza_only_df(articles_df):
    gaza_df = articles_df[articles_df['Title'].str.contains(r'Gaza.*|Palestin.*',\
                            case=False, regex=True)]
    gaza_only_df = gaza_df[~gaza_df['Title'].str.contains(r'Israel.*|IDF*.',\
                            case=False, regex=True)]
    
    return gaza_only_df


# Calculate how frequently terms of interest appear in titles among keywords
def calculate_percentage_of_term(israel_only_df, hamas_only_df, gaza_only_df,\
                                 palestine_only_df, term):
    count_term_israel = israel_only_df['Title'].str.contains(fr'\b{term}.*\b',\
                        case=False, regex=True).sum()
    count_term_hamas = hamas_only_df['Title'].str.contains(fr'\b{term}.*\b',\
                        case=False, regex=True).sum()
    count_term_gaza = gaza_only_df['Title'].str.contains(fr'\b{term}.*\b',\
                        case=False, regex=True).sum()

    total_titles_israel = len(israel_only_df)
    total_titles_hamas = len(hamas_only_df)
    total_titles_gaza = len(gaza_only_df)

    percent_term_israel = (count_term_israel / total_titles_israel) * 100 if\
                            total_titles_israel > 0 else 0
    percent_term_hamas = (count_term_hamas / total_titles_hamas) * 100 if\
                            total_titles_hamas > 0 else 0
    percent_term_gaza = (count_term_gaza / total_titles_gaza) * 100 if\
                            total_titles_gaza > 0 else 0

    return percent_term_israel, percent_term_hamas, percent_term_gaza


def calculate_percentages_of_terms(articles_df, terms):
    percentages = {'Israel': [], 'Hamas': [], 'Gaza': []}

    for term in terms:
        percent_israel, percent_hamas, percent_gaza = calculate_percentage_of_term\
                             (israel_only_df, hamas_only_df, gaza_only_df, term)
        percentages['Israel'].append(percent_israel)
        percentages['Hamas'].append(percent_hamas)
        percentages['Gaza'].append(percent_gaza)
    
    return percentages


# Titles separated by keyword need to be converted to lists before the word cloud 
# can be created
def get_titles_from_df(df, column_name='Title'):
    
    return df[column_name].tolist()


def get_hamas_titles(df):
    
    return get_titles_from_df(df)


def get_israel_titles(df):
    
    return get_titles_from_df(df)


def get_gaza_titles(df):
    
    return get_titles_from_df(df)


def create_wordcloud(titles, additional_stopwords=None):
    text_for_wc = ' '.join(titles)
    
    stopwords = set(STOPWORDS)
    if additional_stopwords:
        stopwords.update(additional_stopwords)
    
    wc = WordCloud(max_words=25,
                   width=2000,
                   height=1000,
                   colormap='tab20c',
                   background_color='#F2EDD7FF',
                   stopwords=stopwords, 
                   random_state=42).generate(text_for_wc)
    
    return wc


# custom_stopwords need to be a global variable here so that it can be read
# by other scripts and passed through create_wordcloud function
custom_stopwords = {'Live updates', 'White', 'Israel', 'Hamas', 'Gaza', \
                    'Netanyahu', 'Biden', 'American', 'White House', 'amid', \
                    'Americans', "Hamas'", 'U', 'S', 'House', 'Live', 'updates', \
                    'Oct'}
    

if __name__ == "__main__":
    os.chdir('/Users/kieranmartin/Documents/DAP Capstone Project')

    articles_df = filter_articles()
    articles_df = filter_news_outlets(articles_df)
    articles_df = analyze_article_sentiments(articles_df)
    df_tfidf_time = process_articles_for_tfidf(articles_df)
    
    israel_only_df = get_israel_only_df(articles_df)
    hamas_only_df = get_hamas_only_df(articles_df)
    gaza_only_df = get_gaza_only_df(articles_df)

    hamas_titles = get_hamas_titles(hamas_only_df)
    israel_titles = get_israel_titles(israel_only_df)
    gaza_titles = get_gaza_titles(gaza_only_df)

    
    

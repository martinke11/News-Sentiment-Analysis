#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 10:41:30 2023

@author: kieranmartin
"""
from newsapi import NewsApiClient
import pandas as pd
import subprocess
import os
    
# Initialize News Article API Client
def init_newsapi():
    newsapi = NewsApiClient(api_key='')
    
    return newsapi


def retrieve_news_articles(from_date, end_date, major_sources, query, newsapi):
    articles_data = []  

    while from_date < end_date:
        to_date = from_date + pd.Timedelta(days=1)

        all_articles = newsapi.get_everything(q=query,
                                              language='en',
                                              sources=major_sources,
                                              from_param=from_date.strftime('%Y-%m-%d'),
                                              to=to_date.strftime('%Y-%m-%d'))

        for article in all_articles['articles']:
            title = article['title']
            date_published = article['publishedAt']

            articles_data.append({
                'Date Published': date_published,
                'News Outlet': article['source']['name'],
                'Title': title
            })

        from_date = to_date

    article_title_df = pd.DataFrame(articles_data)
    
    return article_title_df


# Automatically store all csv files created after each pull in folder in github repo 
def save_dataframe_to_csv_and_push(new_df):
    new_csv_path = os.path.join('data', new_csv_file_name)
    new_df.to_csv(new_csv_path, index=False)
    commit_message = f'Added {new_csv_file_name}'
    subprocess.run(['git', 'add', new_csv_file_name])
    subprocess.run(['git', 'commit', "-m", commit_message])
    # code below to automatically push to repo with SSH key otherwise will have 
    # to go into github desktop and click 'push'
    subprocess.run(["git", "push"])


# Merge all current files in folder in repo 
def merge_csv_files(data, csv_folder_path):
    merged_df = pd.DataFrame()

    for csv_file in data:
        file_path = os.path.join(csv_folder_path, csv_file) 
        df = pd.read_csv(file_path)  
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df


def get_clean_articles_df():
    csv_folder_path = 'data'
    csv_files = os.listdir(csv_folder_path)
    csv_files = [file for file in csv_files if file.endswith('.csv')]
    
    merged_df = merge_csv_files(csv_files, csv_folder_path)
    merged_df = merged_df[merged_df['News Outlet'] != '[Removed]']
    merged_df = merged_df.drop_duplicates(subset='Title')

    return merged_df

    
if __name__ == "__main__":
    os.chdir('/Users/Kieranmartin/Documents/DAP Capstone Project')
    newsapi = init_newsapi()
    new_csv_file_name = 'pull_12_2.csv'
    csv_folder_path = 'data'
    new_csv_file_path = os.path.join(csv_folder_path, new_csv_file_name)
    
    # Prevent making new pull when re-runnng this script if pull was already 
    # completed using the name of the last csv file created.
    if not os.path.exists(new_csv_file_path):
        query = 'Israel OR Hamas OR Gaza' 
        from_date = pd.to_datetime('2023-11-07')
        end_date = pd.to_datetime('2023-12-05') 
        major_sources = 'associated-press, cnn, fox-news, nbc-news, cbs-news,\
                        abc-news, politico, reuters, bloomberg,\
                        the-washington-post, the-wall-street-journal, npr,\
                        usa-today, msnbc, bbc-news'
        pull_sources = retrieve_news_articles(from_date, end_date,\
                        major_sources, query, newsapi)
        save_dataframe_to_csv_and_push(pull_sources)
    else:
        print(f"File {new_csv_file_name} already exists. Skipping article retrieval.")

    articles_df = get_clean_articles_df()


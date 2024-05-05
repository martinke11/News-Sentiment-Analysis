#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:57:50 2024

@author: kieranmartin
"""

import pandas as pd
import os
import text_proccessing as tp
import statsmodels.api as sm
from scipy import stats


# Assign dummy variables to political leaning to do analysis
political_leaning_mapping = {
    'Far Right': 5,
    'Right': 4,
    'Centrist': 3,
    'Left': 2,
    'Far Left': 1
}

# Create a new column 'Political Leaning Numeric' based on the mapping
def analyze_subjectivity_on_political_leaning(articles_df):
    x = articles_df['Political Leaning Numeric']
    y = articles_df['subjectivity']
    x = sm.add_constant(x)

    pl_model = sm.OLS(y, x).fit()
    pl_model_summary = pl_model.summary()
    
    return pl_model, pl_model_summary


def analyze_subjectivity_difference(hamas_only_df, israel_only_df, alpha=0.05):
    avg_subj_hamas = hamas_only_df['subjectivity'].mean()
    avg_subj_israel = israel_only_df['subjectivity'].mean()
    
    t_stat, p_value = stats.ttest_ind(hamas_only_df['subjectivity'],\
                            israel_only_df['subjectivity'], equal_var=False)

    significance = "significant" if p_value < alpha else "not significant"
    
    return avg_subj_hamas, avg_subj_israel, t_stat, p_value, significance


def analyze_subjectivity_by_outlet(hamas_only_df, israel_only_df):
    average_subjectivity_hamas_by_outlet = hamas_only_df.groupby('News Outlet')\
        ['subjectivity'].mean()
    average_subjectivity_israel_by_outlet = israel_only_df.groupby('News Outlet')\
        ['subjectivity'].mean()

    combined_subjectivity = pd.DataFrame({
        'Average Subjectivity Hamas': average_subjectivity_hamas_by_outlet,
        'Average Subjectivity Israel': average_subjectivity_israel_by_outlet
    })

    combined_subjectivity.reset_index(inplace=True)

    return combined_subjectivity


def transform_and_melt_subjectivity(combined_subjectivity):
    # Multiply 'Average Subjectivity Israel' by -1 for for diverging barchart axis
    combined_subjectivity['Average Subjectivity Israel'] *= -1
    df_long = combined_subjectivity.melt(id_vars='News Outlet',\
              var_name='Category', value_name='Subjectivity Score')
    df_long.sort_values(by=['Category', 'Subjectivity Score'], inplace=True)
    
    return df_long


terms = ['Terror', 'Aid', 'Attack', 'Support']

# In each dataframe find terms specified in titles, divide by the number
# of titles in each dataframe and return the percentages of each term
def calculate_percentage_of_term(israel_only_df, hamas_only_df, gaza_only_df,\
                                 term):
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


# Create a dictionary of each group with the percentages of each term found
# throughout their titles to prepare for plotting
def calculate_percentages_of_terms(articles_df, terms):
    percentages = {'Israel': [], 'Hamas': [], 'Gaza': []}
    
    for term in terms:
        percent_israel, percent_hamas, percent_gaza = calculate_percentage_of_term(
                             israel_only_df, hamas_only_df, gaza_only_df, term)
        percentages['Israel'].append(percent_israel)
        percentages['Hamas'].append(percent_hamas)
        percentages['Gaza'].append(percent_gaza)
    
    return percentages

# Citation: ChatGPT

if __name__ == "__main__":
    os.chdir('/Users/kieranmartin/Documents/DAP Capstone Project')

    articles_df = tp.filter_articles()
    articles_df = tp.filter_news_outlets(articles_df)
    articles_df = tp.analyze_article_sentiments(articles_df)
    
    israel_only_df = tp.get_israel_only_df(articles_df)
    hamas_only_df = tp.get_hamas_only_df(articles_df)
    gaza_only_df = tp.get_gaza_only_df(articles_df)
    
    articles_df['Political Leaning Numeric'] = articles_df['Political Leaning']\
                                                .map(political_leaning_mapping)
    political_lean_model_summary = analyze_subjectivity_on_political_leaning(articles_df)
    
    avg_subj_hamas, avg_subj_israel, t_stat, p_value, significance =\
        analyze_subjectivity_difference(hamas_only_df, israel_only_df)

    # Analyze subjectivity by outlet
    combined_subjectivity = analyze_subjectivity_by_outlet(hamas_only_df, israel_only_df)
    df_long = transform_and_melt_subjectivity(combined_subjectivity)
    percentages = calculate_percentages_of_terms(articles_df, terms)






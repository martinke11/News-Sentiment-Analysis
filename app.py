#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:07:38 2024

@author: kieranmartin
"""
import pandas as pd
from textblob import TextBlob
import time
import numpy as np
import warnings
import httpx
import nltk
import os
import spacy
from collections import Counter
import spacy
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as px
import plotly.io as pio
import os
os.chdir('/Users/kieranmartin/Documents/DAP Capstone Project')
import dash
from dash import dcc
from dash import html
from dash import Dash, dcc, html  # Updated imports
import plotly.express as px
import pandas as pd
import plotly.express as px
import pandas as pd
import os
from plotly.subplots import make_subplots

# Run the command below in bash
# kieranmartin$ python /Users/kieranmartin/Documents/DAP Capstone Project/app.py

# Initialize the Dash app
app = Dash(__name__)


articles_df = pd.read_csv('articles_df.csv')

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

israel_only_df = get_israel_only_df(articles_df)
hamas_only_df = get_hamas_only_df(articles_df)
gaza_only_df = get_gaza_only_df(articles_df)

def get_titles_from_df(df, column_name='Title'):
    
    return df[column_name].tolist()


def get_hamas_titles(df):
    
    return get_titles_from_df(df)


def get_israel_titles(df):
    
    return get_titles_from_df(df)


def get_gaza_titles(df):
    
    return get_titles_from_df(df)

hamas_titles = get_hamas_titles(hamas_only_df)
israel_titles = get_israel_titles(israel_only_df)
gaza_titles = get_gaza_titles(gaza_only_df)

terms = ['terror', 'Aid', 'war', 'killed']

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

# Call function from analysis.py that create percentages to plot them
percentages = calculate_percentages_of_terms(articles_df, terms)

def analyze_subjectivity_by_outlet(hamas_only_df, israel_only_df):
    average_subjectivity_hamas_by_outlet = hamas_only_df.groupby('News Outlet')['subjectivity'].mean()
    average_subjectivity_israel_by_outlet = israel_only_df.groupby('News Outlet')['subjectivity'].mean()

    combined_subjectivity = pd.DataFrame({
        'Average Subjectivity Hamas': average_subjectivity_hamas_by_outlet,
        'Average Subjectivity Israel': average_subjectivity_israel_by_outlet
    })

    combined_subjectivity.reset_index(inplace=True)

    return combined_subjectivity


def transform_and_melt_subjectivity(combined_subjectivity):
    # Multiply 'Average Subjectivity Israel' by -1 for diverging barchart axis
    combined_subjectivity['Average Subjectivity Israel'] *= -1
    df_long = combined_subjectivity.melt(id_vars='News Outlet', var_name='Category',\
                                         value_name='Subjectivity Score')
    df_long.sort_values(by=['Category', 'Subjectivity Score'], inplace=True)
    
    return df_long

# Assuming hamas_only_df and israel_only_df are defined and formatted correctly
combined_subjectivity = analyze_subjectivity_by_outlet(hamas_only_df, israel_only_df)

# Transform the combined subjectivity data
df_long = transform_and_melt_subjectivity(combined_subjectivity)

articles_df_filtered = articles_df[articles_df['Group'] != 'Other']
custom_stopwords = {
    'live', 'updates', 'white', 'israel', 'hamas', 'gaza', 'netanyahu', 'biden',
    'american', 'white house', 'amid', 'americans', 'hamas', 'u', 's', 'house',
    'live', 'updates', 'oct'
}
# Get standard English stop words from CountVectorizer and convert to a list
english_stop_words = list(CountVectorizer(stop_words='english').get_stop_words())

# Combine the custom stop words with the standard English stop words
stop_words = list(custom_stopwords) + english_stop_words

# Initialize CountVectorizer with combined stop words
vectorizer = CountVectorizer(stop_words=stop_words)

# Keep the top 25 words for each group
top_words = 25
word_freq = []

# Process titles for each group and count word frequencies
for group in ['Gaza', 'Hamas', 'Israel']:
    # Filter the DataFrame by group
    group_titles = articles_df_filtered[articles_df_filtered['Group'] == group]['Title']
    # Get word counts for the group
    word_counts = vectorizer.fit_transform(group_titles)
    words = vectorizer.get_feature_names_out()
    counts = word_counts.sum(axis=0).A1
    freq_dist = sorted(dict(zip(words, counts)).items(), key=lambda x: x[1], /
                       reverse=True)[:top_words]
    # Add the top words and their frequencies to the list, tagged by group
    for word, count in freq_dist:
        word_freq.append({'Group': group, 'Word': word, 'Frequency': count})

# Create a DataFrame with the word frequencies
word_freq_df = pd.DataFrame(word_freq)

# Sort the DataFrame by group and frequency for clarity
word_freq_df.sort_values(['Group', 'Frequency'], ascending=[True, False], inplace=True)

################################################################################
# time series
articles_df['Date Published'] = pd.to_datetime(articles_df['Date Published']).dt.date
grouped_titles = articles_df.groupby('Date Published')['Title'].apply(lambda x: ' '.join(x))
counts = CountVectorizer(strip_accents='unicode', lowercase=True, stop_words='english', max_features=2000) 
X_counts = counts.fit_transform(grouped_titles)

tf_trans = TfidfTransformer(use_idf=False)
X_tfidf = tf_trans.fit_transform(X_counts)

# Convert to DataFrame
cols = counts.get_feature_names_out()
df_tfidf_time = pd.DataFrame(X_tfidf.todense(), columns=cols, index=grouped_titles.index)

term = 'cease'
# Extract the time series for the term
term_series = df_tfidf_time[term]
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(term_series.index, term_series.values, label=f'TF-IDF of "{term}"', width=0.5)

# Setting up the x-axis with dates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability

# Adding labels and title
ax.set_title(f'Time Series of TF-IDF Scores for "{term}<br><sup>subtitle</sup>"')
ax.set_xlabel('Date Published')
ax.set_ylabel('TF-IDF Score')
ax.legend()

from dash import dcc, html, Input, Output, callback
def filter_by_term(df, term):
    if term in df.columns:
        return df[term]
    else:
        return pd.Series([0] * len(df), index=df.index)



@callback(
    Output('time-series-graph', 'figure'),
    Input('term-dropdown', 'value')
)

def update_graph(selected_term):
    
    term_series = filter_by_term(df_tfidf_time, selected_term.lower())

    # Create the figure using graph objects for more customization
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=term_series.index, y=term_series.values, mode='lines+markers', 
                   name='TF-IDF Score', line=dict(color='navy')),
        secondary_y=False,
    )

    # Update layout for the x-axis
    fig.update_xaxes(
        tickangle=-45,  # rotate the ticks by 90 degrees
        tickmode='linear',  # set the mode of the ticks (auto, linear, array)
        #tickformat='%Y-%m-%d',  # set the format of the ticks
        tickvals=term_series.index[::1],  # set the values at which ticks on this axis appear
        title_text='Date Published'
    )
    
    tickvals = [i/100 for i in range(0, 101, 5)]  # Generate a list from 0 to 1 by 0.05 increments
    ticktext = [f'{i}%' for i in range(0, 101, 5)]  
    
    # Update layout for the y-axis
    fig.update_yaxes(
        tickvals=tickvals,  # Set custom tick values
        ticktext=ticktext,  # Set custom tick labels
        title_text='Term Frequency',  # Axis title
        secondary_y=False
    )
    # fig.update_yaxes(title_text='Term Frequency', secondary_y=False)
    # print("Dates in term_series.index:", term_series.index)
    # print("Dates and events in significant_events:", significant_events)
    
    for date_str, event in significant_events.items():
        # Convert string to date object
        date = pd.to_datetime(date_str).date()
        
        if date in term_series.index:
            # Adjust the ax value for the specific event on 2023-11-21
            ax = -150 if date_str == '2023-11-21' else 20
            ay = 40 if date_str == '2023-11-21' else 80

            fig.add_annotation(
                x=date,
                y=term_series.loc[date] + 0.002,  # Accessing the value
                text=event,
                showarrow=True,
                arrowhead=1,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12, color='white'), 
                ax=ax,
                ay=-ay
            )
            fig.add_annotation(
                x=date,
                y=term_series.loc[date],  # Actual position
                text=event,
                showarrow=True,
                arrowhead=1,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=12, color='black'),  # Main text in black
                ax=ax,
                ay=-ay
            )

    # Update the rest of the layout
    fig.update_layout(
        title_text=f'Key Topics Change in How Frequently they are Mentioned<br><sup>Frequency of "{selected_term}" and changes over time across Major US news outlets</sup>'
    )
    fig.update_layout(
        title=dict(
            #x=0.5,  # Center the title horizontally
            y=0.90,  # Adjust the title vertical position; increase the value to move it higher
            #xanchor='center',
            #yanchor='top'
            font=dict(
                size=24,
            )
        ),
    )

    return fig

#outline in white
significant_events = {
    '2023-10-28': '5000 Palestinians killed as Israel<br>anounces Increased Aerial attacks',
    '2023-11-05': 'Mass Protests emerge across<br>the globe calling for Ceasefire',
    '2023-11-10': 'Israel begins air striking<br>Al-Shifa Hospital.<br>Over 11,000 Palestinians<br>have been killed.',
    '2023-11-21': 'Israels Raid on<br>Al-Shifa Hostpital<br>found no Hamas Command Center,<br>3 Gaza Hopsitals Request<br>Assistance Evacuating',
    '2023-11-22': 'Israels Cabnnet approves<br>hostage deal.<br>Over 15,000 Palestinians<br>have been killed'

}


terms = ['Terror', 'Aid', 'Support']
import plotly.graph_objects as go

def plot_term_percentages_plotly(percentages, terms):
    percentages['Israel'] = [p / 100 for p in percentages['Israel']]
    percentages['Hamas'] = [p / 100 for p in percentages['Hamas']]
    percentages['Gaza'] = [p / 100 for p in percentages['Gaza']]

    # ... your existing code ...


    x = terms
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=percentages['Israel'],
        name='Israel',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=x, y=percentages['Hamas'],
        name='Hamas',
        marker_color='orange'
    ))
    fig.add_trace(go.Bar(
        x=x, y=percentages['Gaza'],
        name='Gaza',
        marker_color='green'
    ))
    fig.update_yaxes(tickformat="0.0%")
    fig.update_layout(
        title='Hamas titles often contained "Terror" terms, whereas<br>"Aid" and "Support" were more common in titles about Israel.<br><sup>Percentages of Terms in Titles by Group</sup>',
        xaxis_tickfont_size=14,
        
        # yaxis=dict(
        #     title='Percentages',
        #     titlefont_size=16,
        #     tickfont_size=14,
        # ),
        legend=dict(
            orientation='h',
            x=0.04,
            y=1.1,
            bgcolor='rgba(255, 255, 255, 0)',

        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.update_layout(
        autosize=False,  # Disable autosizing, use fixed size
        width=700,  # Width of the figure in pixels
        height=600,  # Height of the figure in pixels
        title=dict(
            y=0.97,
            font=dict(size=18),
        ),
    )

    return fig

fig2 = plot_term_percentages_plotly(percentages, terms)


# subtitle: Mean subjectivity scores between News Titles that mention 'Israel' 
# alone and 'Hamas' alone, by news outlet. 
def create_and_display_diverging_chart(df_long):
    custom_order = ['CNN', 'MSNBC', 'NBS News', 'Fox News', 'The Washington Post',
                    'Politico', 'CBS News', 'BBC News', 'ABC News', 'Associated Press', 
                    'The Wall Street Journal', 'Reuters', 'The Hill']
    # Ensure all outlets are in the custom order list
    df_long = df_long[df_long['News Outlet'].isin(custom_order)]

    # Sort the DataFrame by the custom order
    df_long['News Outlet'] = pd.Categorical(df_long['News Outlet'], /
                                            categories=custom_order, ordered=True)
    df_long.sort_values('News Outlet', inplace=True)

    color_discrete_map = {'Average Subjectivity Hamas': 'purple', 
                          'Average Subjectivity Israel': 'green'}

    # Create the bar chart with the specified color map
    fig = px.bar(df_long, y='News Outlet', x='Subjectivity Score', 
                 color='Category',
                 color_discrete_map=color_discrete_map,  # Apply the color map here
                 orientation='h',
                 title="Articles Focusing on Hamas are More Subjective than Those Focusing on Israel.<br><sup>Mean subjectivity scores from SpaCy Natural Language Processing Analysis between News Titles that mention 'Israel' alone and 'Hamas' alone, by news outlet. </sup>",
                 labels={'Subjectivity Score': 'Absolute Average Subjectivity Score'},
                 height=600)
    fig.update_layout(
        title=dict(
            #x=0.5,  # Center the title horizontally
            y=0.97,  # Adjust the title vertical position; increase the value to move it higher
            #xanchor='center',
            #yanchor='top'
            font=dict(
                size=24,
            )
        ),
        # ... your existing layout updates ...
    )
    # Customizing the x-axis with specified ticks
    ticks = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 
             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig.update_xaxes(tickvals=ticks, ticktext=[str(abs(x)) for x in ticks])

    
    fig.add_annotation(
        x=0.03,  # Center
        y=-1.3,  # Position the annotation just below the last bar
        # Text with arrows on both sides
        text="← More Subjective                      More Subjective ⟶",  
        showarrow=True,
        xanchor="center",
        #yanchor="bottom",
        ay=0,
        font=dict(size=12)
    )
    # Adjusting the range of the x-axis
    fig.update_layout(xaxis_range=[-1, 1],
                      legend_orientation='h',
                      legend_x=-0.04, legend_y=1.1)

    return fig

fig1 = create_and_display_diverging_chart(df_long)


def create_treemap(word_freq_df):
    fig = px.treemap(word_freq_df, path=['Group', 'Word'], values='Frequency',
                      title='Positive Words Appear more Frequently in News Titles Mentioning Isreal<br><sup>Top 25 Words by Frequency for Israel, Hamas, and Gaza News Titles</sup>')
    fig.update_layout(
        autosize=False,  # Disable autosizing, use fixed size
        width=1000,  # Width of the figure in pixels
        height=600,  # Height of the figure in pixels
        title=dict(
            y=0.9,
            font=dict(size=24),
        ),
    )
  
    return fig


fig3 = create_treemap(word_freq_df)

app.layout = html.Div(children=[
    html.H1(
        'Natural Language Processing Analysis of Major US New Titles Covering Israel-Hamas War Since Oct. 23 2023', 
        style={
            'textAlign': 'left',
            'marginTop': '20px',
            'font-family': 'Arial, sans-serif',  
            'color': '#2a3f5f',  
            'fontSize': '30px'  
            }
    ),

    html.Div([
        # Column for the graph
        html.Div([
            dcc.Graph(id='subjectivity-graph', figure=fig1)
        ], style={'width': '85%', 'display': 'inline-block'}),  
        
        # Column for the text field
        html.Div([
            #html.Label('Analysis Summary:'),
            dcc.Textarea(
                id='summary-text',
                placeholder="When separating major US News article titles released after October 25th by those that mention Israel only and those that mention Hamas only, on average, titles mentioning Hamas only are more subjective. The difference in subjectivity is significant.\n\n"
                    "Welch's T-Test Results:\n"
                    "Average Subjectivity Across (Hamas): 0.2326\n"
                    "Average Subjectivity (Israel): 0.1716\n"
                    "T-statistic: 2.1325\n"
                    "P-value: 0.0344\n\n"
                    "The P-value of 0.034 is less than the common alpha level of 0.05, indicating that the probability of observing such a difference (or more extreme) by random chance is low. Hence, we can reject the null hypothesis that there is no difference in subjectivity between the two groups.",
                    style={
                        'width': '100%',
                        'height': 450,
                        'font-family': 'Arial, sans-serif',  
                        'font-size': '15px',
                        'color': '#2a3f5f',  # Font color set to black to match the plots
                        'border': 'none',  # Remove the border
                        'box-shadow': 'none',  # Remove box shadow if any
                        'outline': 'none',
                        'margin-top': '110px'# Adjust as needed to match the plot text size
                    },  # You can adjust the height as needed
                    readOnly=True
                ),
            # dcc.Textarea(id='input-text', placeholder='Enter text...', style={'width': '100%', 'height': 100}),
        ], style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),
    # Row for the two columns
    html.Div([
        # Column 1
        html.Div([
            dcc.Graph(id='word-frequency-treemap', figure=fig3)
        ], style={'width': '65%', 'display': 'inline-block'}),  # 50% width of the row

        # Column 2
        html.Div([
            dcc.Graph(id='term-analysis-graph', figure=fig2)
        ], style={'width': '35%', 'display': 'inline-block'}),  # 50% width of the row
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(id='time-series-graph'),
            ], style={'width': '90%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='term-dropdown',
                options=[{'label': term, 'value': term} for term in ['Cease', 'Hostage', 'Aid', 'Terror','Protest', 'Hospital']],
                value='Cease'  # default value
            ),
        ], style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top': '40px'}),
        
    ])
])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


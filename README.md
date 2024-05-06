# News Sentiment Analysis

Screencapture of News Sentiment Analysis Plotly Dashboard running locally:


[![Plotly Dashboard Running Locally](https://img.youtube.com/vi/4A--nK6pMNI/0.jpg)](https://www.youtube.com/watch?v=4A--nK6pMNI)



Research Question:
How do the most popular new outlets in the US differ in the way they present Israel versus Hamas and Palestine in terms of subjectivity and the kinds of words they use to describe each group. Media portrayal can influence public opinion and policy, making it crucial to understand these narratives.

Methodology:
I decided to focus on News Article headlines, since they are instrumental in engaging potential readers. I used a free API from NewsAPI.org to collect thousands of titles that matched the following relevant key words: ‘Israel’, ‘Hamas’, ‘Palestine’, or ‘Gaza’. This API allowed me to specify the language, select particular news sources, and set a date range for the articles. The major constraint of using the free account was that it only provided access to the headlines, excluding the article bodies. Although the API offered short summaries, these were not substantially more informative than the headlines themselves.

A significant limitation was the API's restriction on retrieving news titles only from the past month. This prevented me from accessing headlines from critical periods, such as the week of October 8th, which followed the onset of hostilities. Additionally, while the API included a wide range of major news sources, it notably lacked coverage from The New York Times. Another challenge was the API's limitation on the volume of data retrievable within a certain timeframe—I could only extract a few hundred articles every 12 hours. Ideally, a single data pull would have been sufficient to start analysis; however, I had to perform multiple pulls to accumulate a substantial dataset.

Data Collection:
To systematically manage the collected data, I set up a designated folder within a GitHub repository to store CSV files generated from each data pull. The workflow comprised several custom functions:

retrieve_news_articles function: This was designed to query the API and capture the returned data as a dataframe, adhering to predefined parameters.

save_dataframe_to_csv_and_push function: Post-data retrieval, this function saved the dataframe to a CSV file, committed it to the GitHub repository, and performed an automated push. To facilitate this automation, I generated an SSH key for GitHub and transitioned from HTTPS to SSH repository URLs.

For purposes of reproducible research, the automation of the push operation can be bypassed. By commenting out the subprocess.run(['git', 'push']) line, which I've annotated in the code, users can opt to manually push changes using GitHub Desktop since the CSV file will already be committed to the local repository.

merge_csv_files function: This function consolidates all the CSV files located in the repository's folder into a single dataframe, rendering it ready for subsequent analysis.

To efficiently manage multiple scripts that process the same data frame, I transformed data_wrangling.py, text_processing.py, and analysis.py into Python modules. This structure enables the text processing module to directly call a function from data_wrangling.py to obtain the processed data frame. This approach eliminates the need for intermediate CSV file generation and storage, which would otherwise require text_processing.py to import the CSV file anew. This streamlined method is also applied within the analysis.py, plotting.py, and app.py scripts.

By establishing this modular pipeline, the entire process, from data cleaning to text processing and analysis, is fully automated. To update the Shiny dashboard with additional articles or data only requires adjusting the date parameters for the API requests and executing the data_wrangling.py script. 

I encountered some challenges with text processing. The API lacks parameters to control context, which resulted in retrieving off-topic titles that I used keyword to filter out. For instance, in dealing with articles about an MMA fighter named ‘Israel’, I had to exclude titles that contained his last name and other unrelated terms. Additionally, the API searches for specified keywords in both the title and body text, but I only had access to the titles. I had to filter out titles that did not match my keywords.

The regular expression (regex) I utilized is distinctive because it also filters out titles containing ‘Israel-Hamas,’ a term typically found in the phrase ‘Israel-Hamas war’. Such mentions skewed the sentiment analysis, as the phrase ‘Israel-Hamas war’ alone was being coded as significantly subjective. For example, a title reading “Israel-Hamas war: Live Updates” would receive a subjectivity score of 0.5, although it did not contextually describe Israel or Hamas. However, the regex retains titles like “Israel-Hamas war rages as outcry grows over Gaza crisis” because it includes a relevant keyword beyond the term ‘Israel-Hamas’.

Analysis and Results:
After a comprehensive text processing phase for cleaning, I applied text processing techniques in various ways for different analyses. Initially, I utilized the entire dataframe of articles to perform a TF-IDF (Term Frequency-Inverse Document Frequency) analysis of all the titles, which I then grouped by date. This allowed me to observe the temporal changes in mentions of specific topics or phrases.

I created separate dataframes to analyze mentions of 'Israel-only'—containing titles that referred to Israel or the IDF without mentioning 'Hamas'—and 'Hamas-only'—with titles mentioning 'Hamas' but not 'Israel.' The goal was to objectively compare the subjectivity in titles mentioning either Israel or Hamas. Following this, I compared the mean subjectivity scores between the two groups, both individually and by news outlet. I also examined the frequency of certain words for each group and generated corresponding word clouds. Hamas titles had the words ‘Terror” and ‘Terrorist” more frequently in their titles, while Israel had the words “Aid” and “Support” more frequently mentioned in their titles. 

I used the Ad Fontes Media Bias Chart to divide my news outlets into 5 categories of political bias: Far Left, Left, Centrist, Right, and Far Right to see if political leaning could predict subjectivity. These results showed a statistically significant relationship where moving from left to right made titles more subjective. However, this relationship is very weak, as this model only explains 1.1% of the variability in subjectivity scores. 

Moreover, a Welch’s T-Test was conducted to determine if titles mentioning 'Hamas' were more subjective than those mentioning 'Israel.' The Welch’s T-Test results show that news titles mentioning 'Hamas' are more subjective, with an average subjectivity score of 0.233, than those mentioning 'Israel', which have an average score of 0.172. The difference is statistically significant, with a T-Statistic of 2.132 and a p-value of 0.034.

For future research, enhancing the robustness of these results would require a broader dataset, including dates from October 7th onwards, incorporating sources like the New York Times, and expanding the title corpus. This, however, would necessitate access to a more advanced, albeit more costly, version of the API.


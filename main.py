from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import tweepy
import re
import pandas as pd
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
from io import BytesIO



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator






li = []
cnt_neg = 0
cnt_neu = 0
cnt_pos = 0
li_pos = []
li_neg = []
li_neu = []
likes_pos = 0
likes_neg=0
likes_neu = 0
like_cnt={'positive':likes_pos,'neutral':likes_neu,'negative':likes_neg}

cnt = {"positive":cnt_pos,"neutral":cnt_neu,"negative":cnt_neg}


def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    r = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\ / \ / \S+)", " ", tweet).split())
    r = ' '.join(re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet).split())
    no_punct = ""
    for char in r:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def get_tweet_sentiment(tweet):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''

    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment
    if analysis.sentiment.polarity > 0:
        # like_cnt['positive']+=int(tweet.favorite_count)
        li_pos.append(clean_tweet(tweet.lower()))
        cnt['positive']+=1
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        # like_cnt['neutral'] +=int(tweet.favorite_count)

        li_neu.append(clean_tweet(tweet.lower()))
        cnt["neutral"] += 1
        return 'neutral'
    else:
        # like_cnt['negative'] +=int(tweet.favorite_count)

        li_neg.append(clean_tweet(tweet.lower()))
        cnt['negative'] += 1
        return 'negative'


def search_for_hashtags(consumer_key, consumer_secret, access_token, access_token_secret, hashtag_phrase):
    # create authentication for accessing Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # initialize Tweepy API
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # get the name of the spreadsheet we will write to
    fname = '_'.join(re.findall(r"#(\w+)", hashtag_phrase))

    with open('%s.csv' % (fname), 'w',encoding="utf-8") as file:
        w = csv.writer(file)

        # write header row to spreadsheet
        w.writerow(
            ['tweet_text', 'username','user_created_at', 'created_at', 'all_hashtags', 'followers_count', 'Location', 'No: of Likes', "Re-Tweets",'user_description',"longitude","latitude",
             'Sentiment'])
        # w.writerow(['tweet_text', 'username', 'all_hashtags', 'followers_count'])
        # for each tweet matching our hashtags, write relevant info to the spreadsheet
        for tweet in tweepy.Cursor(api.search, q=hashtag_phrase + ' -filter:retweets',
                                   lang="en", tweet_mode='extended').items(100):
            longitude = None
            latitude = None
            if tweet.coordinates:
                longitude = tweet.coordinates['coordinates'][0]
                latitude = tweet.coordinates['coordinates'][1]

            if get_tweet_sentiment(tweet.full_text.replace('\n', ' ')) == 'positive':
                like_cnt['positive'] += int(tweet.favorite_count)

            if get_tweet_sentiment(tweet.full_text.replace('\n', ' ')) == 'negative':
                like_cnt['positive'] += int(tweet.favorite_count)

            if get_tweet_sentiment(tweet.full_text.replace('\n', ' ')) == 'neutral':
                like_cnt['neutral'] += int(tweet.favorite_count)



            if tweet.user.location == "India" or tweet.user.location == "INDIA":
                li.append(clean_tweet(tweet.full_text.lower()))
                w.writerow((tweet.full_text.replace('\n', ' ').encode('utf-8'), tweet.user.screen_name.encode('utf-8'), tweet.user.created_at, tweet.created_at
                            ,[e['text'] for e in tweet._json['entities']['hashtags']], tweet.user.followers_count,
                            tweet.user.location, tweet.favorite_count, tweet.retweet_count,(tweet.user.description),longitude,latitude,
                            get_tweet_sentiment(tweet.full_text.replace('\n', ' '))))

    # MultiPlot(li)
    return li


def noise_removal(noise):
    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(noise)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def display(x,pos):
    clean_text = []
    for i in x:
        clean_text.append(noise_removal(i))

    words = get_all_words(clean_text)
    ## Creating FreqDist for whole BoW, keeping the 10 most common tokens
    all_fdist = FreqDist(words)
    fd = pd.DataFrame(all_fdist.most_common(10),
    columns=["Word", "Frequency"]).reindex()



    ## Plotly  plotting
    if pos == 0:
        fig = px.bar(fd, x="Word", y="Frequency", title='All words')
        fig.update_traces(marker_color='rgb(240,128,128)',
                          marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.8)
        fig.show()

    if pos == 1:

        fig = px.bar(fd, x="Word", y="Frequency", title='Positive Words')
        fig.update_traces(marker_color='rgb(240,128,128)',
                          marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.8)
        fig.show()

    if pos == 2:
        fig = px.bar(fd, x="Word", y="Frequency", title='Neutral Words')
        fig.update_traces(marker_color='rgb(240,128,128)',
                          marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.8)
        fig.show()

    if pos == 3:
        fig = px.bar(fd, x="Word", y="Frequency", title='Negative Words')
        fig.update_traces(marker_color='rgb(240,128,128)',
                          marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.8)
        fig.show()




def MultiPlot(x):
    clean_text = []
    for i in x:
        clean_text.append(noise_removal(i))

    words = get_all_words(clean_text)
    ## Creating FreqDist for whole BoW, keeping the 10 most common tokens
    all_fdist = FreqDist(words)
    fd = pd.DataFrame(all_fdist.most_common(10),
                      columns=["Word", "Frequency"]).reindex()

    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    # fig = make_subplots(
    #     rows=2, cols=2,
    #     column_widths=[1, 0.4],
    #     row_heights=[0.6, 0.4],
    #     specs=[[{"type": "scatter", "rowspan": 2},
    #             {"type": "choropleth"}],
    #            [None, {"type": "bar"}]])
    # fig.add_trace(go.Bar(x=fd["Word"], y=fd["Frequency"],
    #                      name="Freq Dist"), row=2, col=2)
    #
    # fig.update_traces(marker_color='rgb(59, 89, 152)',
    #                   marker_line_color='rgb(8,48,107)',
    #                   marker_line_width=0.5, opacity=0.7, row=2, col=2)

    val=cnt.values()
    key=cnt.keys()
    dict1={'Sentiment':key,'Vals':val}
    fig1=pd.DataFrame.from_dict(dict1)
    val1 = like_cnt.values()
    key1 = like_cnt.keys()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(key1), y=list(val1),
                             mode='lines',
                             name='Count of likes'))


    fig.add_trace(go.Bar(x=fig1['Sentiment'], y=fig1['Vals'],
                         name="Tweet Polarity Count"))

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],

    )


    return fig


def WordCloud1(clean_text):
    s = ''


    for i in clean_text:
        for j in i:

            s+=' '+str(j)

    wordcloud = WordCloud().generate(s)
    return wordcloud.to_image()



def dashapp(list2):
    fig = MultiPlot(list2)
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    clean_text = []
    for i in list2:
        clean_text.append(noise_removal(i))




    app.layout = html.Div(
        style={'background':colors['background']},children=[
        html.H1(
            children='Hello There!',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.Div([
            html.H6(children="Change the value in the text box to your new #Search", style={
                'textAlign': 'center',
                'color': colors['text']
            })
        ]),

        html.Div([
            html.Div(children="Input: ", style={
                'color': colors['text']}),
            dcc.Input(id='my-input-1', type='text', value=''),
            html.Button(id='my-input', n_clicks=0, children='Submit')
        ]),

        html.H3(id='output',children='Tweet sentiment on #yoga', style={
            'textAlign': 'center',
            'color': colors['text']
        }),



        dcc.Graph(
            id='my-output',
            figure=fig
        ),


        html.Br(),

        html.Div([
            html.H6(children="Word Cloud", style={
                'textAlign': 'left',
                'color': colors['text']
            })
        ]),

        html.Div([
            html.Img(id="image_wc",src=WordCloud1(clean_text)),
        ]),

    ])

    @app.callback(
      Output(component_id='my-output', component_property='figure'),
        Input('my-input', 'n_clicks'),
        State('my-input-1', 'value'))
    def update_output_div(n, input):

        chh = '#' + input

        list1 = search_for_hashtags("uiJXzxgVIsmQL43M3pbId4Lt1", "PHoV0Rcgo8g7WNvqx5xrW0gsPgtSDzjgFzhL5FZ1JVMHpvg6IW",
                                    "1183066418444521472-Fer4LX1WGWbdbHPkC5yoZIIsF0EHn2",
                                    "oaAxBjvPSNq4JI7Gi6Mq5KGb0qY1NNyJC1o5Tq0yHFnbj", chh)



        return MultiPlot(list1)


    @app.callback(
        Output('output','children'),
        Input('my-input', 'n_clicks'),
        State('my-input-1', 'value')
    )
    def Update(click,val):
        if click==0:
            sentence='Tweet sentiment on #Yoga'
        else:
            sentence='Tweet sentiment on #'+val
        return sentence


    @app.callback(
        Output('image_wc','src'),
        Input('my-input', 'n_clicks'),
        State('my-input-1', 'value')

    )

    def cloud(val,val1):

        word='#'+val1

        list3=search_for_hashtags("uiJXzxgVIsmQL43M3pbId4Lt1", "PHoV0Rcgo8g7WNvqx5xrW0gsPgtSDzjgFzhL5FZ1JVMHpvg6IW",
                    "1183066418444521472-Fer4LX1WGWbdbHPkC5yoZIIsF0EHn2",
                    "oaAxBjvPSNq4JI7Gi6Mq5KGb0qY1NNyJC1o5Tq0yHFnbj", word)

        clean_text = []
        for i in list3:
            clean_text.append(noise_removal(i))


        img = BytesIO()
        WordCloud1(clean_text).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())




    if __name__ == '__main__':
        app.run_server()


dashapp(search_for_hashtags("uiJXzxgVIsmQL43M3pbId4Lt1", "PHoV0Rcgo8g7WNvqx5xrW0gsPgtSDzjgFzhL5FZ1JVMHpvg6IW",
                    "1183066418444521472-Fer4LX1WGWbdbHPkC5yoZIIsF0EHn2",
                    "oaAxBjvPSNq4JI7Gi6Mq5KGb0qY1NNyJC1o5Tq0yHFnbj", "#yoga"))








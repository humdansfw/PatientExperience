#patientexperiencepipeline.py>
# coding: utf-8



#import main tools we will most likely use -----------------------------------------------------------

import regex as re

import numpy as np

import pandas as pd

#LDA prep and bigrams
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

#lemmatization
import spacy
import stanza
'''Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020. [pdf][bib]'''
import spacy_stanza
stanza.download("en")
nlp = spacy_stanza.load_pipeline("en")

#plotting
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()

#Natural Language Toolkit
import nltk; 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist

#coherence visualization
import matplotlib.pyplot as plt

# sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for the 'vadersenter' function
'''VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool. 
        It is fully open-sourced under the [MIT License] (we sincerely appreciate all attributions and readily accept most contributions, but please don't hold us liable).

        Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
        Sentiment Analysis of Social Media Text. Eighth International Conference on
        Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
        
        pip install vaderSentiment
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        #note: depending on how you installed (e.g., using source code download versus pip install), you may need to import like this:
        #from vaderSentiment import SentimentIntensityAnalyzer'''


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


# Global lists for use in functions

patient_experience_stoppers = ['from', 'subject', 're', 'edu', 'use', 'put', 'make', 'need', 'say', 'address', 'get', 'also', 'come', 'liz'] # extends stopwords.



# Define functions used in pipeline

def load_data(filename):
    '''
    Create function to: 
        read data into a pandas dataframe
        dataframe is named df
    Parameters:
        filename: string
    Returns:
        df: dataframe
    '''
    try:
        if (filename.lower().find(".csv") >= 0):
            df = pd.read_csv(filename)
        elif (filename.lower().find(".xlsx") >= 0):
            df = pd.read_excel(filename)
    except:
        print("ERROR: File not found.")
        return
    return df


def clean_data(data):
    '''
    Takes in a text or series of texts and passes data through list comprehensions that clean the data for analysis.
    Parameters:
        data: list of texts (strings)
    Returns:
        data: cleaned version
    '''
    
    # Convert objects to strings and lowercase
    data = [str(sent).lower() for sent in data]
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\n+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    # Deaccentuate the words
    data = [gensim.utils.deaccent(sent) for sent in data]
    
    return data


def sent_to_words(sentences):
    '''
    Uses gensim to convert the sentences to lists of words while also removing punctuations
    Parameters:
        sentences: list
    Returns:
        list: lists of words
    '''
    
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
        
def remove_stopwords(texts):
    '''
    list comprehension to only return words that are not stopwords
    Parameters:
        texts: list of cleaned word lists
    Returns:
        texts: now without stopwords
    '''
    
    stop_words = stopwords.words('english')
    stop_words.extend(patient_experience_stoppers)

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    '''
    List comprehension applying the bigram_mod function to the texts
    Parameters:
        texts: list of word lists.
    Returns:
        texts: now with bigrams.
    '''
    
    # Build the bigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''
    Instantiate empty list; texts_out
    For loop for sentence in texts
        apply nlp to sentence
        append lemmatized token to empty list if we allowed its part-of-speech tag (using spacy .lemma_)
    Returns:
        texts_out list
    '''
    
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def word_list(lemmatized_words):
    '''
    Takes a list of lists of words and takes a count of all of the words in the parent list
    Parameters:
        list: list of word lists.
    Returns:
        list: sentences.
    '''
    allWords = []
    for wordList in lemmatized_words:
        allWords += wordList
    
    freq = FreqDist(allWords)

    freqsdf = pd.DataFrame(list(freq.items()), columns = ["Word","Frequency"]) #create dataframe
    freqsdf = freqsdf[['Frequency','Word']] #swap column positions for export to word cloud
    freqsdf.sort_values(by='Frequency', ascending=False, inplace=True) #sort in descending order of frequency
    
    
    #Output the frequency distribution for visualization (e.g. word-cloud).
    name = input('Choose a suffix for the wordfreqsSUFFIX.csv file' )
    freqsdf.to_csv(f'wordfreqs{name}.csv', index=False, header=False)
   
    return freqsdf
    
    freqsdf.head(10) #check the top ten.

    
    
    
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute u_mass coherence for n number of topics

    Parameters:
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : int, Max number of topics
        start: int, what number of topics to start with
        step: int, step this many
    Returns:
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for numt in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=numt)
        model_list.append(model)
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v') #change coherence parameter to either c_v or u_mass
        coherence_values.append(cm.get_coherence())

    return model_list, coherence_values

def word2sents(listofwordlists): # takes in the list we made that contains each comment separated into a list of words
    '''
    Takes a list containing lists of words and puts the word lists back into sentences
    Parameters:
        listofwordlists: list
    Returns:
        newsents: list
    '''
    
    newsents = [] # we will populate this new list with full sentences
    for sentence in listofwordlists:
        sentence = ' '.join(sentence) # lets join our words
        newsents.append(sentence) # lets append our full sentence to our empty list
    return newsents # return this list so we can assign it to a new variable/name


def sbl(df, data):

    cleanlist = clean_data(data) # bringing in a column of patient experience dataframe as a list and cleaning that list
    wordlist = list(sent_to_words(cleanlist)) # split sentences into word lists
    stoppedlist = remove_stopwords(wordlist) #
    lemmalist = lemmatization(stoppedlist, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return lemmalist
        
def clean_df(df):
    '''
    This will take the dataframe pulled in with load_data and clean each of the columns but will not lemmatize, bigram, remove stopwords.
    Parameters:
        df: patient experience dataframe
    Returns:
        df: patient experience dataframe cleaned
    '''
    
    columns = ['Best Part', 'Worst Part', 'Suggestions']
    
    for column in columns:
        df[column] = clean_data(df[column])
    
    return df

def vadersenter(sentences):
    '''
    Takes list of sentences and computes polarity scores on each sentence. Gets used in vader_df function.
    Parameters:
        sentences: list
    Returns:
        scores: list
     '''   
    
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence) # polarity_scores thrwing error "float object is not iterable" solved with data_words in above cells
        scores.append(analyzer.polarity_scores(sentence)['compound'])
    return scores

def vader_df(df):
    '''
    Takes the previous vadersenter function and calls it over the comment columns to return polarity scores in new dataframe columns.
    Parameters:
        df: dataframe with cleaned columns.
    Returns:
        df: dataframe with added columns for polarty scores.
    '''


    best_scores = vadersenter(list(df['Best Part'])) 
    worst_scores = vadersenter(list(df['Worst Part']))
    sugg_scores = vadersenter(list(df['Suggestions']))

    df['Best Part Compound Polarity'] = best_scores
    df['Worst Part Compound Polarity'] = worst_scores
    df['Suggestions Compound Polarity'] = sugg_scores  
    columns = ['Best Part Compound Polarity', 'Worst Part Compound Polarity', 'Suggestions Compound Polarity']
    df['Average Score'] = (df[columns].sum(axis=1))/3

def quarterclouds(df):
    '''
    Takes dataframe and creates quarterly word clouds from data. The user will specify the year that the function will take the quarters of. In the future, this can become a function that takes a sliding window that can cover for example, June 1 2021 - May 30 2022. This will require more user input.
    Parameters: 
        df: Pandas dataframe with Datetime column as well as at least the 'Best Part' column.
    Returns:
        Four csv files. One for each quarter. 
    '''
    year = input('For what year would you like to view the quarters')
    dfquarterly = df[['Date','Best Part']] # Creating new dataframe from just these two columns. For PR purposes, we will not use the worst parts or suggestions.
    
    dfq1 = dfquarterly.loc[df['Date'].between(f'{year}-01-01',f'{year}-03-31')].dropna() # We are manually selecting quarters based on specific dates.
    dfq2 = dfquarterly.loc[df['Date'].between(f'{year}-04-01',f'{year}-06-30')].dropna() # With .between(leftbound, rightbound, inclusive), the parameter inclusive is set to true
    dfq3 = dfquarterly.loc[df['Date'].between(f'{year}-07-01',f'{year}-09-30')].dropna() # by default.
    dfq4 = dfquarterly.loc[df['Date'].between(f'{year}-10-01',f'{year}-12-31')].dropna() # Dropping NA values so that we dont cloud our word clouds.
    
    dfquarters = [dfq1, dfq2, dfq3, dfq4] # list of our quarters to iterate over
    
    count = 1 #initialize the counter for our file names
    for dfq in dfquarters: # for each dataframe quarter in the dataframe quarters list:
        listq = dfq['Best Part'].tolist() # list quarter is created from turning the column to list
        listq = clean_data(listq) # run the clean_data function over our list
        wordlq = list(sent_to_words(listq)) # word-list-quarter is created from using the sent_to_words function on our list.
        stoppedq = remove_stopwords(wordlq) # use remove_stopwords function on our list.
        lemmaq = lemmatization(stoppedq, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) # lemmatize our list.
        freqq = word_list(lemmaq) # use word_list function on the list to get frequency distribution.
        #freqq.to_csv('fq %d .csv' %count, index=False, header=False) # print each freq dist to a csv file for wordcloud generation.
        freqq.to_csv(f'FrequencyByQuarter{count}.csv', index=False, header=False)
        count = count+1 # increase the count by one so that we have an output file written for each quarter in our for loop.
    
    return lemmaq # returning cleaned lemma list for testing out LDA prep function.

def datetimestamp(df):
    '''
    Create the Day, Time, and Timestamp(int64) columns from the Date column.
    '''
    tsdf = df
    tsdf['Time'] = tsdf['Date'].dt.time
    for time in tsdf['Time']:
        time = time.strftime("%-j")
    tsdf['Day'] = tsdf['Date'].dt.date
    tsdf['Timestamp'] = tsdf['Time'].values.astype(np.int64) // 10 ** 9
    
    return tsdf



def LDApipe(df):
    '''
    Takes in a dataframe. Based on the user's choice, the function will run the desired ~column~ through the preparation for the LDA model and outputs the LDA models topics in an intertopic distribution map.
    Parameters:
        df: dataframe with "Best Part", "Worst Part", and "Suggestions" columns
    Returns:
        vis: Visualized intertopic distribution map.
    '''
    # collect user input for which column they would like to use
    column = input('Which column would you like to pass through the LDA model? "Best Part", "Worst Part", or "Suggestions"? ').lower() # normalize user input
    
    if 'best' in column:
        column = 'Best Part'
        print('\nYou chose the "Best Part" column. If this is incorrect, run again with exact spelling.\n')
        print(df[column].head())
    elif 'worst' in column:
        column = 'Worst Part'
        print('\nYou chose the "Worst Part" column. If this is incorrect, run again with exact spelling.\n')
        print(df[column].head())
    elif 'sug' in column:
        column = 'Suggestions'
        print('\nYou chose the "Suggestions" column. If this is incorrect, run again with exact spelling.\n')
        print(df[column].head())
    else:
        print('\nThat didnt work. Try again.')
        column = None # Let's hope it doesn't come to this.
        
    cleanlist = clean_data(df[column]) # bringing in a colun of patient experience dataframe as a list and cleaning that list
    wordlist = list(sent_to_words(cleanlist)) # split sentences into word lists
    stoppedlist = remove_stopwords(wordlist) #
    lemmalist = lemmatization(stoppedlist, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) #
    bgrms = make_bigrams(lemmalist) #
    id2word = corpora.Dictionary(bgrms) # mapping between words and their integer ids.
    texts = bgrms #
    corpus = [id2word.doc2bow(text) for text in texts] #Convert ~document~ (a list of words) into the ~bag-of-words~ format = list of (token_id, token_count) 2-tuples.
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=30, step=5) # Instantiate compute_coherence_values
       
    maxco = max(coherence_values) # finding the max coherence with coherence values function
    
    # Plotting the coherence scores and number of topics
    
    limit=30; start=2; step=5; # instantiate start limit and step according to compute_coherence_values
    x = range(start, limit, step) # set x-axis
    tops = list(x)
    cv = {'Topics':tops, 'Score':coherence_values}
    cv = pd.DataFrame(cv)
    optimal_topics = cv['Topics'][cv['Score'].idxmax()]
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, # Instantiate lda_model
                                           id2word=id2word,
                                           num_topics=optimal_topics, # Passing in the numt argument which specifies the optimal number of topics provided by the LDAprep function
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word) # Instantiate pyLDAvis
    print('For every topic, two probabilities p1 and p2 are calculated.\nP1 – p(topic t / document d) = the proportion of words in document d that are currently assigned to topic t.\nP2 – p(word w / topic t) = the proportion of assignments to topic t over all documents that come from this word w.')
    
    return vis
    
    




    
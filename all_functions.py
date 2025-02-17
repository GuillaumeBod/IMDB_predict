import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from string import digits
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
import nltk
from string import digits
nltk.download('stopwords')



def get_html_from_link(page_link):
    '''
        Get HTML from web page and parse it.

        :param page_link: link of the webpage we want to scrap
        :type page_link: string
        :return: BeautifulSoup object (HTML parsed)
        :rtype: bs4.BeautifulSoup
    '''
    response = requests.get(page_link)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def collecte_rating(soup):
    '''
        Extract movies ratings from web page and parse it.

        :param soup: BeautifulSoup object (HTML parsed)
        :type soup: bs4.BeautifulSoup
        :return: df_rating : dataframe with ratings
        :rtype: DataFrame
    '''
    movies_rating = []
    for element in soup.find_all('td', {"class" : "ratingColumn imdbRating"}):
        movies_rating.append(element.text)

    df_rating = pd.DataFrame({"movies_ratings" : movies_rating})
    df_rating = df_rating.replace('\n','', regex=True)
    return df_rating

def collecte_titles_links(soup):
    '''
        Extract movies titles and links from web page and parse it.

        :param soup: BeautifulSoup object (HTML parsed)
        :type soup: bs4.BeautifulSoup
        :return: df_titles : dataframe with movies_titles and movies_links
        :rtype: DataFrame
    '''
    url = "https://www.imdb.com/"
    movies_titles = []
    movies_links = []
    for element in soup.find_all('a'):
        ref = element.get("href")
        if ref == None:
            print(None)
        elif '/title/' in ref:
            movies_titles.append(element.text)
            movies_links.append(url+ref)

    movies_titles = [np.nan if x == ' \n' else x for x in movies_titles]

    df_titles = pd.DataFrame({"movies_titles": movies_titles, "movies_links": movies_links})
    df_titles = df_titles.dropna().reset_index()
    df_titles = df_titles.drop([0], axis=0)
    df_titles = df_titles.drop("index", axis = 1)

    return df_titles

def extract_movie_info(movie_html):
    '''
        Extract movie info from movie_html

        :param movie_html: BeautifulSoup Element that contains all movies links
        :type movie_html: bs4.BeautifulSoup
        :return: DataFrame with the names of the DirectorS, Writters, actors, titles for each movies
        :rtype: DataFrame
    '''

    # TODO : get book_title, book_rating and book_author from book_html and return this tuple
    infos = [k.text for k in movie_html.find_all('div', {'class' : 'credit_summary_item'})]
    titles = movie_html.find('h1').text

    df = {"directeur":infos[0].split("\n")[2].split(','),
          "scenaristes":infos[1].split("\n")[2].split(','),
          "vedettes":infos[2].split("\n")[2].split(','),
          "titres" : titles}


    return df

def extract_budget(movie_html):
    '''
        Extract movie budget from movie_html

        :param movie_html: BeautifulSoup Element that contains all movies links
        :type movie_html: bs4.BeautifulSoup
        :return: the budget of a movie
        :rtype: int
    '''
    budget = 0
    for div in movie_html.find_all('div', {'class' : 'txt-block'}):
        
        if 'Budget' in  div.text:
            budget = int(''.join([a for a in div.text.strip() if a.isdigit()]))

    return(budget)

def extract_country(movie_html):
    '''
        Extract movie country from movie_html

        :param movie_html: BeautifulSoup Element that contains all movies links
        :type movie_html: bs4.BeautifulSoup
        :return: the country of a movie
        :rtype: String
    '''

    country=[]
    for link in movie_html.find_all("a"):
        try:
            if 'country_of_origin' in  link["href"]:
                country.append(link.text)
        except KeyError:
            pass
    return(country)

def extract_synopsis(movie_html):
    '''
        Extract movie synopsis from movie_html

        :param movie_html: BeautifulSoup Element that contains all movies links
        :type movie_html: bs4.BeautifulSoup
        :return: the synopsis of a movie
        :rtype: String
    '''

    link = movie_html.find("div", {"class": "inline canwrap"})
    synopsis = link.find('span').text.strip()
    return(synopsis)

def extract_all_infos(df):
    '''
        Extract all infos of a movie from a dataframe of links for each movie

        :param df: DataFrame with movies titles and movies links
        :type df: DataFrame
        :return: DataFrame with all informations since to all functions above
        :rtype: DataFrame
    '''

    budget=[]
    synopsis=[]
    country=[]
    directeur = []
    scenaristes = []
    vedettes = []
    titres = []
    
    for url in df["movies_links"]:
        movie_html = get_html_from_link(url)
        
        budget.append(extract_budget(movie_html))
        synopsis.append(extract_synopsis(movie_html))
        country.append(extract_country(movie_html))
        infos = extract_movie_info(movie_html)
        directeur.append(infos["directeur"])
        scenaristes.append(infos["scenaristes"])
        vedettes.append(infos["vedettes"])
        titres.append(infos["vedettes"])
    df["budget"]=budget
    df["synopsis"]=synopsis
    df["country"]=country
    df["directeur"] = directeur
    df["scenaristes"] = scenaristes
    df["vedettes"] = vedettes
    df["titres"] = titres
    return(df)

nltk_stopwords = set(nltk.corpus.stopwords.words("english"))
sklearn_stowords = set(ENGLISH_STOP_WORDS)
STOP_WORDS = nltk_stopwords.union(sklearn_stowords)

def remove_digits(text):
    '''
        Remove digits from text

        :param text: text like synopsis in our case 
        :type df: String
        :return: text without digits
        :rtype: String
    '''
    remove_digits = str.maketrans('', '', digits)
    res = text.lower().translate(remove_digits)
    return res

stem = nltk.stem.snowball.EnglishStemmer()

def remove_tiny_words(text):
    '''
        Remove words with length lower than 2 from text

        :param text: text like synopsis in our case 
        :type df: String
        :return: text without tiny words
        :rtype: String
    '''

    new_text = ""
    for word in text.split():
        if len(word) > 2:
            new_text = new_text + " " + stem.stem(word)
    return new_text

def remove_digits_tiny(df):
    '''
        Remove words with length lower than 2 and digits from text since to the 2 functions above

        :param df: dataframe with all movies informations
        :type df: DataFrame
        :return: df without digits and tiny words in the column synopsis and movies_titles
        :rtype: DataFrame
    '''

    new_synopsis =[]
    acteurs_scenaristes_directeur=[]
    new_titre = []
    
    df = pd.concat([df, pd.get_dummies(df["vedettes"].apply(pd.Series).stack()).sum(level=0)], axis=1)
    df = pd.concat([df, pd.get_dummies(df["directeur"].apply(pd.Series).stack()).sum(level=0)], axis=1)
    df = pd.concat([df, pd.get_dummies(df["scenaristes"].apply(pd.Series).stack()).sum(level=0)], axis=1)
    df = pd.concat([df, pd.get_dummies(df["country"].apply(pd.Series).stack()).sum(level=0)], axis=1)
    
    for synopsis in df['synopsis']:
        new_synopsis.append(remove_tiny_words(remove_digits(synopsis)))
    for titre in df['movies_titles']:
        new_titre.append(remove_tiny_words(remove_digits(titre)))
    df["new_titre"] = new_titre
    df["new_synopsis"] = new_synopsis
    return(df)

def vectorize_df(df):
    '''
        Vectorize movie DataFrame

        :param df: dataframe with all movies informations
        :type df: DataFrame
        :return: X_vectorized wich is vectorized
        :rtype: DataFrame
    '''
    cv = CountVectorizer(lowercase=True, 
                     stop_words=STOP_WORDS, 
                     ngram_range=(1, 2), 
                     max_features=2000)

    cv.fit(df[["new_synopsis","new_titre"]])

    X_vectorized = cv.transform(df["new_synopsis"])
    X_vectorized = pd.DataFrame(X_vectorized.toarray(), 
                                columns=cv.get_feature_names())

    X_vectorized = pd.concat([X_vectorized, df.drop(["synopsis",
                                                        "movies_titles",
                                                        "movies_links",
                                                        "country",
                                                        "directeur",
                                                        "scenaristes",
                                                        "vedettes",
                                                        "titres",
                                                        "new_titre",
                                                        "new_synopsis"], axis = 1)], axis=1).drop(["new_synopsis",
                                                                                                "new_titre"], axis =1)

    return X_vectorized


def modify_ratings(df):
    ratings = []
    for rating in df["rating"]:
        if type(rating) != 'float':
            one, two = rating.split(",")
            ratings.append(one +"."+two)

    df_ratings = pd.DataFrame({"ratings":ratings})

    df_ratings = df_ratings.astype(np.float16)

    return df_ratings
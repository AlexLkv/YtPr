import csv
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
import pymorphy2

# Download NLTK resources
nltk.download('stopwords')

# Initialize stopwords and morphological analyzer
russian_stopwords = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()

# Function to remove punctuation
def remove_punct(text):
    return re.sub(r'\W+', ' ', text)

# Function to normalize word
def norm_word(w):
    parsed_word = morph.parse(w)[0]
    return parsed_word.normal_form

# Function to filter nouns and adjectives
def is_noun_or_adjective(x):
    parsed_word = morph.parse(x)[0]
    return parsed_word.tag.POS in ('NOUN', 'ADJF')

# URL
url_base = "https://habr.com/ru/"

# Initialize CSV file and write headers
with open('companies.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Name', 'Rating', 'Description', 'Sphere', "txts"])

# Companies data
companies = {'yandex': [793744, 793722, 45678], 'gazprombank': [56558, 5516, 7111], 'ncloudtech': [7141, 7844, 4641],
             'xeovo': [1234, 1235, 1236]}

# Process each company
for company in companies:
    url = url_base + 'companies/' + company + '/profile/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extracting company info
    name = soup.find_all(class_='tm-company-card__name')[0].text.strip()
    rating = soup.find_all(class_='tm-votes-lever__score-counter tm-votes-lever__score-counter tm-votes-lever__score-counter_rating')[0].text.strip()
    description = ''.join(soup.find('span', class_='tm-company-profile__content').find('span').get_text().split('\n')).replace("\n", ' ')
    industries = ''.join(soup.find_all(class_='tm-company-profile__categories')[0].text.strip().split('\n            \n             '))

    # Extracting articles text
    articles = []
    for txt_id in companies[company]:
        url = url_base + 'articles/' + str(txt_id)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        txt = soup.find_all(class_='tm-article-body')[0].text.replace("\n", ' ')
        articles.append(txt)

    # Preprocess data
    description = remove_punct(description.lower())
    description = ' '.join([norm_word(word) for word in description.split() if word not in russian_stopwords])
    description = ' '.join([word for word in description.split() if is_noun_or_adjective(word)])

    #industries = remove_punct(description.lower())
    #industries = ' '.join([norm_word(word) for word in description.split() if word not in russian_stopwords])
    #industries = ' '.join([word for word in description.split() if is_noun_or_adjective(word)])

    articles_text = ' '.join([remove_punct(article.lower()) for article in articles])
    articles_text = ' '.join([norm_word(word) for word in articles_text.split() if word not in russian_stopwords])
    articles_text = ' '.join([word for word in articles_text.split() if is_noun_or_adjective(word)])

    # Writing data to CSV
    with open('companies.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, rating, description, industries, articles_text])

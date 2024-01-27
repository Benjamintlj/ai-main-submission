#######################################################
# Imports
#######################################################
import csv
import nltk
import ssl

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords, wordnet
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize


#######################################################
# Download any necessary resources
#######################################################
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


#######################################################
# Init lemmatizer and stopwords
#######################################################
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


#######################################################
# Methods
#######################################################
def response_tone(user_message, answers):
    """
    Gets the tone of the users message, then responds with a similar tone.
    :param user_message: The users message.
    :param answers: A tuple of answers: [0] - neutral, [1] - negative, [2] - positive.
    :return: A response matching the users tone.
    """
    text_blob = TextBlob(user_message)
    sentiment = text_blob.sentiment.polarity

    if sentiment > 0:
        response = answers[2]
    elif sentiment < 0:
        response = answers[1]
    else:
        response = answers[0]

    return response


def get_wordnet_part_of_speech(part_of_speech):
    """
    Gets the wordnet POS for a given treebank tag.
    :param part_of_speech: Treebank POS.
    :return: Wordnet POS.
    """
    tags = {
        'R': wordnet.ADV,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'J': wordnet.ADJ
    }

    return tags.get(part_of_speech[0], wordnet.NOUN)


def lemmatisation_tokenizer(question):
    """
    Tokenizes a string and removes any stopwords, symbols and then lemmatizes each token.
    :param question: Any string to be manipulated.
    :return: Filtered a tokens list.
    """
    tokens = word_tokenize(question)
    pos_tags = pos_tag(tokens)

    filtered_tokens = []
    for token, part_of_speach in pos_tags:
        # remove none alphanumeric tokens
        if 0 < len(token) < 2 and not token.isalnum():
            continue

        # remove stopwords
        if token in stop_words:
            continue

        # Lemmatize
        filtered_tokens.append(lemmatizer.lemmatize(token, get_wordnet_part_of_speech(part_of_speach)))

    return filtered_tokens


def get_most_similar_question_and_answer(users_question):
    """
    Finds the most similar question to passed question.
    :param users_question: The user's question to compair against the csv.
    :return: The most valid answer.
    """
    with open('./learn-files/question_and_answer.csv') as file:
        reader = csv.reader(file)
        next(reader)  # skip headers

        csv_questions = []
        csv_answers = []
        for row in reader:
            csv_questions.append(row[0])
            csv_answers.append((row[1], row[2], row[3]))

        # Get the tf*idf of each token
        vectorizer = TfidfVectorizer(tokenizer=lemmatisation_tokenizer)
        csv_questions_tf_idf = vectorizer.fit_transform(csv_questions)

        # Compare each csv question to the users question
        user_question_tf_idf = vectorizer.transform([users_question])
        cosine_sim = cosine_similarity(user_question_tf_idf, csv_questions_tf_idf)[0]

        # Get the most similar question
        highest_index = None
        highest_similarity = 0.5  # Anything below this the computer does not know
        for index, similarity in enumerate(cosine_sim):
            if similarity > highest_similarity:
                highest_index = index
                highest_similarity = similarity

        if highest_index is None:
            response = 'I am sorry, I do not know'
        else:
            response = response_tone(users_question, csv_answers[highest_index])

        return response

import nltk
import pickle
import re
import numpy as np
import csv
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'POST_EMBEDDINGS_FOLDER': 'post_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data\word_embeddings.tsv',
}


def format_text(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    embeddings = {}
    with open(embeddings_path, newline='') as embedding_file:
        reader = csv.reader(embedding_file, delimiter='\t')
        for line in reader:
            word = line[0]
            embedding = np.array(line[1:]).astype(np.float32)
            embeddings[word] = embedding
        dim = len(line) - 1
    return embeddings, dim

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    words_embedding = [embeddings[word] for word in question.split() if word in embeddings]
    # replace embedding with zeros if the embedding is not available
    if not words_embedding:
        return np.zeros(dim)
    words_embedding = np.array(words_embedding)
    return words_embedding.mean(axis=0)

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

		
class PostRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.post_embeddings_folder = paths['POST_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.post_embeddings_folder, tag_name + ".pkl")
        post_ids, post_embeddings = unpickle_file(embeddings_path)
        return post_ids, post_embeddings

    def get_best_post(self, question, tag_name):
        """ Returns id of the most similar post for the question.
            The search is performed across the posts with a given tag.
        """
        post_ids, post_embeddings = self.__load_embeddings_by_tag(tag_name)

        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_post = pairwise_distances_argmin( X=question_vec.reshape(1, self.embeddings_dim), Y=post_embeddings, metric='cosine')
        
        return post_ids[best_post][0]


class DialogueManager(object):
    def __init__(self, paths):
        print("Initialization...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think the question is about %s\nThis post might help you: https://stackoverflow.com/questions/%s'

        # Stackoverflow part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.post_ranker = PostRanker(paths)

        # Create chatbot
        self.create_chatbot()

    def create_chatbot(self):
        """Creates chat bot and train it in English."""

       
        self.chatbot = ChatBot(
            'fifluisa',
            trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
        )
        self.chatbot.train("chatterbot.corpus.english")
        self.chatbot.set_trainer(ListTrainer)

       
    def generate_answer(self, question):
        """Combines Normal and stackoverflow conversations parts using intent recognition."""

        # Classify question if chat is just normal dialogue or related to a stackoverflow question.
         
        prepared_question = format_text(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Normal dialogue  part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(prepared_question)
            return response
        
        # Stackoverflow part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]

            # Pass prepared_question to post_ranker to get predictions.
            post_id = self.post_ranker.get_best_post(prepared_question, tag)
           
            return self.ANSWER_TEMPLATE % (tag, post_id)
		
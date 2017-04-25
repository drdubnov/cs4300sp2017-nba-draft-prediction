from .models import Docs
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import nltk

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

tfidf = pickle.load(open(os.path.join(BASE_DIR, "data", 'model.pkl')))
prospect_docs = pickle.load(open(os.path.join(BASE_DIR, "data", 'prospects.pkl'))).toarray()
player_docs = pickle.load(open(os.path.join(BASE_DIR, "data", 'players.pkl'))).toarray()

prospect_to_position = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_position.json")))
player_to_position = json.load(open(os.path.join(BASE_DIR, "data","player_to_position.json")))

ind_to_prospect = json.load(open(os.path.join(BASE_DIR, "data", "ind_to_prospect.json")))
ind_to_player = json.load(open(os.path.join(BASE_DIR, "data", "ind_to_player.json")))

#prospect_to_ind = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_ind.json")))
#player_to_ind = json.load(open(os.path.join(BASE_DIR, "data", "player_to_ind.json")))

prospect_to_image = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_image.json")))
prospect_to_prob = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_prob.json")))
prospect_to_sentences = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_sentences.json")))



def find_similar_players(prospect_ind, k=3):
	prospect_name = ind_to_prospect[str(prospect_ind)]
	prospect_position = prospect_to_position[prospect_name]
	prospect_doc = prospect_docs[prospect_ind]
	sims = []
	for ind, row in enumerate(player_docs):
	    name = ind_to_player[str(ind)]
	    position = player_to_position[name]
	    if not set(position).isdisjoint(prospect_position):
	        doc = row.flatten()
	        if not np.all(doc == 0.0):
	            dotted = np.dot(doc, prospect_doc)	
	            sim = dotted/(np.linalg.norm(doc)*np.linalg.norm(prospect_doc))
	            sims.append((name, sim))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims[:k]


def find_similar(q, pos, num_keywords=5, num_sentences=3):
	transformed = tfidf.transform([q]).toarray().flatten()
	if np.all(transformed == 0.0):
		return ["Query is out of vocabulary"]
	sims = []
	for ind, row in enumerate(prospect_docs):
		prosp = ind_to_prospect[str(ind)]
		position = prospect_to_position[prosp]
		if pos == "any" or pos.upper() in position:
			prosp_image = prospect_to_image[prosp]
			doc = row.flatten()
			dotted = np.dot(doc, transformed)
			if not np.all(doc == 0.0) and not np.all(dotted == 0):
				mult = np.multiply(doc, transformed)
				num_matched = np.size(np.where(mult != 0))
				top_words_inds = np.argsort(mult)[::-1][:min(num_keywords, num_matched)]
				top_words = [tfidf.get_feature_names()[top_word_ind] for top_word_ind in top_words_inds]

				sentences = prospect_to_sentences[prosp]
				sentences_with_top_words = []
				actual_top_words = []
				sentences_with_top_words_cosine_sim = []

				for sentence in sentences:
					tokens = nltk.word_tokenize(sentence.lower())
					top_words_in_sentence = []
					for word in top_words:
						if word in tokens:
							top_words_in_sentence.append(word)

					if len(top_words_in_sentence) > 0:
						sentence_tfidf = tfidf.transform([sentence]).toarray().flatten()
						sentence_cosine_sim = np.dot(transformed, sentence_tfidf) / (np.linalg.norm(transformed) *
				                                                                     np.linalg.norm(sentence_tfidf))
						sentences_with_top_words.append(sentence)
						actual_top_words.append(top_words_in_sentence)
						sentences_with_top_words_cosine_sim.append(sentence_cosine_sim)
				best_sentences = sorted(zip(sentences_with_top_words, actual_top_words, sentences_with_top_words_cosine_sim), 
					key=lambda x: (x[2], np.size(x[1])), reverse=True)[:min(num_sentences, len(sentences_with_top_words))]
				output_sents = [sent[0] for sent in best_sentences]
				sim = dotted/(np.linalg.norm(doc)*np.linalg.norm(transformed))
				if sim > 0.0:
					sims.append((prosp, "{:.3f}".format(sim), find_similar_players(ind), prosp_image, 
						"Probability of NBA Success: {:.3f}".format(prospect_to_prob[prosp]), output_sents, "/".join(position)))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims
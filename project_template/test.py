from .models import Docs
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

tfidf = pickle.load(open(os.path.join(BASE_DIR, "data", 'model.pkl')))
prospect_docs = pickle.load(open(os.path.join(BASE_DIR, "data", 'prospects.pkl'))).toarray()
player_docs = pickle.load(open(os.path.join(BASE_DIR, "data", 'players.pkl'))).toarray()

prospect_to_position = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_position.json")))
player_to_position = json.load(open(os.path.join(BASE_DIR, "data","player_to_position.json")))

ind_to_prospect = json.load(open(os.path.join(BASE_DIR, "data", "ind_to_prospect.json")))
ind_to_player = json.load(open(os.path.join(BASE_DIR, "data", "ind_to_player.json")))

prospect_to_image = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_image.json")))
prospect_to_link = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_link.json")))
prospect_to_prob = json.load(open(os.path.join(BASE_DIR, "data", "prospect_to_prob.json")))
prospect_to_sentences = json.load(open(os.path.join(BASE_DIR, "data", "final_cleaned_prospects_to_sents.json")))
prospect_to_docs = {prosp: "".join(sents) for (prosp, sents) in prospect_to_sentences.items()}
tfidf2 = pickle.load(open(os.path.join(BASE_DIR, "data", 'model2.pkl')))
player_to_tfidf = json.load(open(os.path.join(BASE_DIR, "data", "player_to_tfidf.json")))
label_to_synonyms = json.load(open(os.path.join(BASE_DIR, "data", "label_to_synonyms.json")))


sid = SentimentIntensityAnalyzer()

for p in prospect_to_sentences.keys():
	if p not in prospect_to_link:
		print p, "link"
	if p not in prospect_to_image:
		print p, "image"
	if p not in prospect_to_prob:
		print p, "prob"

def sort_positions(pos):
	pos_to_num = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
	sorted_pos = sorted(pos, key=lambda x:pos_to_num[x])
	return "/".join(sorted_pos)


def find_similar_players(prospect_name, tfidf_vector, k=3):
	prospect_position = prospect_to_position[prospect_name]
	sims = []
	for player, doc in player_to_tfidf.items():
		position = player_to_position[player]
		if not set(position).isdisjoint(prospect_position):
			transformed = np.array(doc)
			if not np.all(transformed == 0.0):
				dotted = np.dot(transformed, tfidf_vector)	
				sim = dotted/(np.linalg.norm(transformed)*np.linalg.norm(tfidf_vector))
				sims.append((player, "{:.3f}".format(sim)))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims[:k]

def find_similar_players_old(prospect_ind, k=3):
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

def find_similar(query, pos, version, num_keywords=5, num_sentences=3):
	if int(version) == 3:
		return find_similar_new(query, pos)
	else:
		return find_similar_old(query, pos, version)

def bold_query(query, outputs):
	out = []
	for sent in outputs:
		bolded = []
		for word in sent.split():
			cleaned = word.lower()
			for punct in ["."]:
				cleaned = cleaned.replace(punct, "")
			if cleaned in query:
				bolded.append("<em><strong>"+word+"</strong></em>")
			else:
				bolded.append(word)
		out.append(" ".join(bolded))
	return out

def find_similar_new(query, pos, num_keywords=5, num_sentences=3):
	new_query = [query]
	for word in query.split():
		if word in label_to_synonyms:
			new_query.extend(label_to_synonyms[word])
	transformed = tfidf2.transform([" ".join(new_query)]).toarray().flatten()
	if np.all(transformed == 0.0):
		return ["Query is out of vocabulary"]
	sims = []
	for prosp, sents in prospect_to_sentences.items():
		position = prospect_to_position[prosp]
		if pos == "any" or pos.upper() in position:
			prosp_image = prospect_to_image[prosp]
			total_doc = np.zeros(len(tfidf2.get_feature_names()))
			doc = prospect_to_docs[prosp]
			mult = np.multiply(tfidf2.transform([doc]).toarray().flatten(), transformed)
			num_matched = np.size(np.where(mult != 0))
			top_words_inds = np.argsort(mult)[::-1][:min(num_keywords, num_matched)]
			top_words = [tfidf2.get_feature_names()[top_word_ind] for top_word_ind in top_words_inds]
			        
			sentences_with_top_words = []
			actual_top_words = []
			sentences_with_top_words_cosine_sim = []
			for sentence in sents:
				tokens = nltk.word_tokenize(sentence.lower())
				top_words_in_sentence = []
				for word in top_words:
				    if word in tokens:
						top_words_in_sentence.append(word)
						if len(top_words_in_sentence) > 0:
							sentence_tfidf = tfidf2.transform([sentence]).toarray().flatten()
							sentence_cosine_sim = np.dot(transformed, 
								sentence_tfidf) / (np.linalg.norm(transformed) * np.linalg.norm(sentence_tfidf))
							sentences_with_top_words.append(sentence)
							actual_top_words.append(top_words_in_sentence)
							sentences_with_top_words_cosine_sim.append(sentence_cosine_sim)
				t_doc = tfidf2.transform([sentence]).toarray().flatten()
				if not np.all(t_doc == 0.0):
				    ss = sid.polarity_scores(sentence.lower())
				    total_doc += t_doc*np.exp(-ss["neg"])
			if not np.all(total_doc == 0):
				sim = np.dot(total_doc, transformed)/(np.linalg.norm(total_doc)*np.linalg.norm(transformed))
				best_sentences = sorted(zip(sentences_with_top_words, actual_top_words, sentences_with_top_words_cosine_sim), 
				                        key=lambda x: (x[2], np.size(x[1])), 
				                        reverse=True)[:min(num_sentences, len(sentences_with_top_words))]
				output_sents = list(set([sent[0] for sent in best_sentences]))
				sims.append((prosp, "{:.3f}".format(sim), find_similar_players(prosp, total_doc), prosp_image, 
					"Probability of NBA Success: {:.3f}".format(prospect_to_prob[prosp]), 
					bold_query(new_query, output_sents), "{} - ".format(sort_positions(position))))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims

def find_similar_old(q, pos, version, num_keywords=5, num_sentences=3):
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
						sentence_cosine_sim = np.dot(transformed, 
							sentence_tfidf) / (np.linalg.norm(transformed) * np.linalg.norm(sentence_tfidf))
						sentences_with_top_words.append(sentence)
						actual_top_words.append(top_words_in_sentence)
						sentences_with_top_words_cosine_sim.append(sentence_cosine_sim)
				best_sentences = sorted(zip(sentences_with_top_words, actual_top_words, sentences_with_top_words_cosine_sim), 
					key=lambda x: (x[2], np.size(x[1])), reverse=True)[:min(num_sentences, len(sentences_with_top_words))]
				output_sents = [sent[0] for sent in best_sentences]
				sim = dotted/(np.linalg.norm(doc)*np.linalg.norm(transformed))
				if sim > 0.0:
					if int(version) == 1:
						sims.append((prosp, "{:.3f}".format(sim), find_similar_players_old(ind), prosp_image, "", [], ""))
					elif int(version) == 2:
						sims.append((prosp, "{:.3f}".format(sim), find_similar_players_old(ind), prosp_image, 
							"Probability of NBA Success: {:.3f}".format(prospect_to_prob[prosp]), 
							output_sents, "{} - ".format("/".join(position))))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims
from .models import Docs
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import re
from scipy import io

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# prospect_docs = io.mmread(os.path.join(BASE_DIR, "data", 'prospects.mtx')).toarray()
# player_docs = io.mmread(os.path.join(BASE_DIR, "data", 'players.mtx')).toarray()

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


def find_similar(q, pos):
	transformed = tfidf.transform([q]).toarray().flatten()
	if np.all(transformed == 0.0):
		return ["Query is out of vocabulary"]
	sims = []
	for ind, row in enumerate(prospect_docs):
		prosp = ind_to_prospect[str(ind)]
		if pos == "any" or pos.upper() in prospect_to_position[prosp]:
			prosp_image = prospect_to_image[prosp]
			doc = row.flatten()
			if not np.all(doc == 0.0):
				dotted = np.dot(doc, transformed)
				sim = dotted/(np.linalg.norm(doc)*np.linalg.norm(transformed))
				if sim > 0.0:
					sims.append((prosp, sim, find_similar_players(ind), prosp_image))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims
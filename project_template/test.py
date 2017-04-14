from .models import Docs
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def read_json(n):
	path = Docs.objects.get(id = n).address
	file = open(path)
	json_file = json.load(file)
	return json_file

prospect_docs = pickle.load(open(os.path.join(BASE_DIR, "project_template", "data", 'prospects.pkl'))).toarray()
player_docs = pickle.load(open(os.path.join(BASE_DIR, "project_template", "data", 'players.pkl'))).toarray()

prospect_to_position = json.load(open(os.path.join(BASE_DIR, "project_template", "data", "prospect_to_position.json")))
player_to_position = json.load(open(os.path.join(BASE_DIR, "project_template", "data","curr_player_to_position.json")))

ind_to_prospect = json.load(open(os.path.join(BASE_DIR, "project_template", "data", "ind_to_prospect.json")))
prospect_to_ind = json.load(open(os.path.join(BASE_DIR, "project_template", "data", "prospect_to_ind.json")))
ind_to_player = json.load(open(os.path.join(BASE_DIR, "project_template", "data", "ind_to_player.json")))
#player_to_ind = json.load(open(os.path.join(BASE_DIR, "project_template", "data", "player_to_ind.json")))

tfidf = pickle.load(open(os.path.join(BASE_DIR, "project_template", "data", 'model.pkl')))


def find_similar_players(prospect, k=2):
    prospect_position = prospect_to_position[prospect]
    prospect_ind = prospect_to_ind[prospect]
    prospect_doc = prospect_docs[prospect_ind]
    sims = []
    for ind, row in enumerate(player_docs):
        name = ind_to_player[str(ind)]
        position = player_to_position[name]
        if position in prospect_position:
            doc = row.flatten()
            if not np.all(doc == 0.0):
                dotted = np.dot(doc, prospect_doc)
                sim = dotted/(np.linalg.norm(doc)*np.linalg.norm(prospect_doc))
                sims.append((name, sim))
    sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
    return sorted_sims[:k]


def find_similar(q, pos):
	transformed = tfidf.transform([q]).toarray().flatten()
	sims = []
	for ind, row in enumerate(prospect_docs):
		prosp = ind_to_prospect[str(ind)]
		if pos == "any" or pos.upper() in prospect_to_position[prosp]:
			doc = row.flatten()
			if not np.all(doc == 0.0):
				dotted = np.dot(doc, transformed)
				sim = dotted/(np.linalg.norm(doc)*np.linalg.norm(transformed))
				sims.append((prosp, sim, find_similar_players(prosp)))
	sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
	return sorted_sims
from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = pd.read_csv('./data for model.csv', encoding= 'unicode_escape')
selected_features = ['Title','netProfit','grossrevenue','reasonForSale','niches','countries']
for feature in selected_features:
    data[feature] = data[feature].fillna('')

combined_features = data['Title']+' '+str(data['netProfit'])+' '+str(data['grossrevenue'])+' '+data['reasonForSale']+' '+data['niches']+' '+data['countries']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)


async def function2(field_name):
    list_of_all_titles = data['niches'].tolist()
    find_close_match = difflib.get_close_matches(field_name, list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = data[data.niches == close_match]['Index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_business = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
    print('businesses suggested for you : \n')
    businessID = []
    new_title = []

    for business in sorted_similar_business[0:10]:
        index = business[0]
        new_title.append(data[data.index==index]['Id'].values[0])

    businessID = list(set(new_title))
    return businessID


@app.get("/getSimilarBusiness/{niche}")
async def read_item(niche: str):
    businessID = await function2(niche)
    print(businessID)
    return {"key" : businessID};
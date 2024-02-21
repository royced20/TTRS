import requests 
from flask import Flask,render_template,request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

tedx_dict = pickle.load(open("tedx.pkl","rb"))
tedx = pd.DataFrame(tedx_dict)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = "english")
vectors = cv.fit_transform(tedx['tags']).toarray()


similarity = pickle.load(open("similarity.pkl","rb"))
def recommend(videos):
    video_index = tedx[tedx['title'] == videos].index[0]  # Get video index
    distances = similarity[video_index]  # Get array of cosine similarity values for the given video
    video_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]  # Return a list of 5 most similar videos

    tedx_title = []
    ted_video = []
    for i in video_list:
       ted_video.append(tedx.iloc[i[0]].url )# Print the url of the 5 most similar videos
       tedx_title.append((tedx.iloc[i[0]].title ).title())# Print the url of the 5 most similar videos


    return  ted_video,tedx_title
def text_similarity(text):
    text_sim = cosine_similarity(cv.transform([text]), vectors)
    video_list = sorted(list(enumerate(text_sim[0])), reverse=True, key=lambda x: x[1])[0:5]

    teds_video = []
    teds_title = []

    for video in video_list:
        teds_video.append(tedx.iloc[video[0]].url)
        teds_title.append((tedx.iloc[video[0]].title).title())

    return teds_video,teds_title
def get_recommendations(watched_video):
    if tedx['title'].eq(watched_video).any():
        return recommend(watched_video.lower())  # Calling the 'recommend' function
    else:
        return text_similarity(watched_video.lower())

app = Flask(__name__)

@app.route('/')

def home():
    return render_template("tedx.html")

@app.route('/predict',methods = ["GET","POST"])

def predict():

    type_input = request.form['type_input']
    users,title = get_recommendations(type_input)

    return render_template("tedx.html",users = users,title = title,values = zip(users,title) )



if __name__ == "__main__":
    app.run(debug = True)

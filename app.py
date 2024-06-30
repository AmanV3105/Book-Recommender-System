import streamlit as st
import numpy as np 
import pickle
st.header("Books Recommender System using Machine Learning")
model = pickle.load(open("artifacts/model.pkl" , "rb"))
books_name = pickle.load(open("artifacts/books_name.pkl" , "rb"))
final_rating = pickle.load(open("artifacts/final_rating.pkl" , "rb"))
rating_pivot = pickle.load(open("artifacts/rating_pivot.pkl" , "rb"))

selected_books = st.selectbox(
    "Type or Select a Book you like!" , 
    books_name
)
def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = [] 
    for book_id in suggestion:
        book_name.append(rating_pivot.index[book_id])
        
    for name in book_name[0]:
        ids = np.where(final_rating["title"] == name)[0][0]
        ids_index.append(ids)
        
    for idx in ids_index:
        url = final_rating.iloc[idx]["img_url"]
        poster_url.append(url)
             
    return poster_url

def recommend_book(book_name):
    book_list = []
    book_index = np.where(rating_pivot.index == book_name)[0][0]
    distance , suggestion = model.kneighbors(rating_pivot.iloc[book_index, :].values.reshape(1, -1) , n_neighbors=6)
    poster_url = fetch_poster(suggestion)
    for i in range(len(suggestion)):
        books = rating_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
            
    return book_list , poster_url

if st.button("Show Recommendation"):
    recommended_book , poster_url = recommend_book(selected_books)
    col1 , col2 , col3 , col4 , col5 = st.columns(5)
    with col1:
        st.text(recommended_book[1])
        st.image(poster_url[1])
    
    with col2:
        st.text(recommended_book[2])
        st.image(poster_url[2])
        
    with col3:
        st.text(recommended_book[3])
        st.image(poster_url[3])
        
    with col4:
        st.text(recommended_book[4])
        st.image(poster_url[4])
         
    with col5:
        st.text(recommended_book[5])
        st.image(poster_url[5])
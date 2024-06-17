import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep.eda import plot

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import nltk
from nltk.tokenize import word_tokenize, punkt
from nltk.corpus import stopwords

import gensim
from gensim.utils import *
from gensim.parsing.preprocessing import *
from gensim import corpora, models, similarities
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity

from surprise import *
from surprise.model_selection.validation import cross_validate

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

#--------------
# GUI
st.title("CourseRec for Coursera")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["HOME", "FIND A COURSE", "ABOUT US", "ABOUT THIS PROJECT"])

with tab1:
    # st.header("Home")
    st.markdown("""
        #### Welcome to CourseRec!
        
        **CourseFinder** is your go-to app for discovering the best Coursera courses tailored to your interests and professional goals. 
        
        ##### How It Works:
        - Simply **enter a keyword** related to the skills or topics you're interested in learning.
        - Hit the **'Find Best Matches'** button to generate a list of recommended courses.
        - Explore course details and select the one that best fits your learning journey.
        
        Start your personalized learning experience with **CourseFinder** today!
    """, unsafe_allow_html=True)


with tab2:
    st.header("Find a Course")
    # Create two columns for user input and button
    col1, col2 = st.columns([3, 1])

    # User input in the first column with a unique key
    with col1:
        user_input = st.text_input("What do you want to learn?")

    # Button in the second column
    with col2:
        find_matches = st.button('Find Best Matches', key="find_matches")

    # 1. Read data
    courses = pd.read_csv("courses.csv", encoding='utf-8')
    reviews = pd.read_csv("reviews.csv", encoding='utf-8')

    # 2. Data pre-processing

    # Drop duplicates
    courses.drop_duplicates(inplace=True)
    reviews.drop_duplicates(inplace=True)

    # Drop unnecessary columns
    courses.drop('CourseID', axis=1, inplace=True)
    reviews.drop('DateOfReview', axis=1, inplace=True)

    # Convert columns of object datatype to string
    courses = courses.astype({col: 'string' for col in courses.select_dtypes('object').columns})
    reviews = reviews.astype({col: 'string' for col in reviews.select_dtypes('object').columns})

    # Handle missing values
    courses['Level'].fillna('', inplace=True)
    courses['Results'].fillna('', inplace=True)
    reviews.dropna(subset=['ReviewContent'], inplace=True)

    # Strip ' level' from Level
    courses['Level'] = courses['Level'].str.replace(' level', '')

    # Strip 'By ' from ReviewerName
    reviews['ReviewerName'] = reviews['ReviewerName'].str.strip('By ')

    # Engineer features
    # Get the list of string columns
    # string_cols = list(courses.select_dtypes(include='string').columns)
    courses['Course'] = courses[['CourseName', 'Unit', 'Level', 'Results']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    courses['ProcessedCourse'] = courses['Course'].apply(strip_punctuation)\
                                              .apply(strip_multiple_whitespaces)\
                                              .apply(strip_non_alphanum)\
                                              .apply(strip_numeric)\
                                              .apply(lower_to_unicode)\
                                              .apply(remove_stopwords)
    
    # Tokenize Course
    courses['TokenizedCourse'] = courses['ProcessedCourse'].progress_apply(word_tokenize)

    # Remove reviews by Deleted A
    reviews = reviews[reviews.ReviewerName != 'Deleted A']


    # 3. Build model

    ## Gensim
    # Create a dictionary representation of the documents
    dictionary = Dictionary(courses['TokenizedCourse'])

    # Create a Corpus: BOW representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in courses['TokenizedCourse']]

    # Use TF-IDF Model to process corpus, obtaining index
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                                num_features = len(dictionary.token2id))

    # Define a function to suggest courses similar to a particular one
    def similar_gensim(course, num=5):
        # Prepocess the course name
        tokens = course.lower().split()
        # Create a bag of words from the course name
        bow = dictionary.doc2bow(tokens)
        # Calculate similarity
        sim = index[tfidf[bow]]
        # Sort similarity in a descending order
        sim = sorted(enumerate(sim), key=lambda item: -item[1])
        # Get names of most similar courses
        results = []
        for x, y in sim:
            if courses.iloc[x]['CourseName'] != course:
                results.append(courses.iloc[x]['CourseName'])
            if len(results) == num:
                break
        return results
    # # Print results
    # print(f"Similar courses to '{course}':\n")
    # for result in results:
    #   print(result)

    # ## SVD
    # reader = Reader(rating_scale=(1, 5))
    # data = Dataset.load_from_df(reviews[['ReviewerName', 'CourseName', 'RatingStar']], reader)

    # # Singular value decomposition
    # algorithm = SVD()

    # # Run 5-fold cross-validation and print results
    # results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # # Fit trainset to the SVD model
    # trainset = data.build_full_trainset()
    # algorithm.fit(trainset)

    # # Define a function to suggest courses to a specific reviewer
    # def similar_svd(name, num=5):
    #   reviewed = reviews[reviews['ReviewerName']==name]['CourseName'].to_list()
    #   results = reviews[['CourseName']].copy()
    #   results['EstScore'] = results['CourseName'].apply(lambda x: algorithm.predict(name, x).est)
    #   results = results.sort_values(by=['EstScore'], ascending=False).drop_duplicates()
    #   results = results[~results['CourseName'].isin(reviewed)]

    #   return results

    #4. Evaluate model

    # Test the model with random courses
    similar_gensim('Davis')
    similar_gensim('machine learning')
    similar_gensim('data science')


    #5. Save models
    # Save Gensim model
    pkl_gensim = "gensim_model.pkl"
    tfidf.save(pkl_gensim)

    # # Save SVD model
    # pkl_svd = "svd_model.pkl"
    # with open(pkl_svd, 'wb') as file:  
    #     pickle.dump(algorithm, file)

    #6. Load models 
    # Load Gensim model
    gensim_model = models.TfidfModel.load(pkl_gensim)

    # # Load SVD model
    # with open(pkl_svd, 'rb') as file:
    #     svd_model = pickle.load(file)

    # Initialize session state for selected courses
    if 'selected_courses' not in st.session_state:
        st.session_state['selected_courses'] = []


    # Define a function to suggest courses similar to user input
    def suggest_courses(user_input):
        suggested_courses = similar_gensim(user_input)
        return suggested_courses

    # Display the suggested courses and their details
    if find_matches:
        suggested_courses = suggest_courses(user_input)
        
        # Iterate over the suggested courses and display their details
        for course_name in suggested_courses:
            # Find the course details
            course_details = courses[courses['CourseName'] == course_name].iloc[0]
            
            # Display the course details using st.write or st.table
            st.markdown(f"<h2 style='font-size:125%;'><b>{course_details['CourseName']}</b></h2>", unsafe_allow_html=True)
            st.write("Description:", course_details['Results'])
            st.write("Provider:", course_details['Unit'])
            st.write("Average Rating:", course_details['AvgStar'])
            st.write("Level:", course_details['Level'])
            
            # You can add a separator for better readability
            st.markdown("---")


with tab3:
    # Create two columns for the profiles
    col1, col2 = st.columns(2)

    with col1:
        # Profile for Phuong N.
        st.subheader("Phuong N.")
        st.write("Email: phuong.n@gmail.com")
        st.write('Phone: +123456789')
        st.write('Role: Model Fine-Tuning Specialist')
        st.write('''
            Phuong played a pivotal role in enhancing the performance of our recommendation system. 
            With a keen eye for detail, Phuong meticulously fine-tuned the model parameters to improve accuracy and ensure the most relevant course suggestions.
        ''')

    with col2:
        # Profile for Linh N.
        st.subheader("Linh N.")
        st.write("Email: linh.n@gmail.com")
        st.write('Phone: +987654321')
        st.write('Role: Data Processing and Modelling Expert')
        st.write('''
            Linh was instrumental in constructing the foundation of our recommendation system. 
            From data collection to preprocessing, Linh ensured the data was clean and structured. 
            Linh's expertise in modeling also contributed significantly to the initial build of our system.
        ''')

    # Add a section for shared responsibilities
    st.write('''
        Both Phuong N. and Linh N. brought their unique strengths to the table in a collaborative effort on the app design. 
        Their combined insights have been invaluable in creating an intuitive and user-friendly interface that enhances the overall user experience.
    ''')


with tab4:
    st.subheader("Objectives")
    st.write('''
        This project is designed to empower users with a robust course recommendation system. 
        By entering any keyword related to skills, expertise, educational institutions, course names, or levels, 
        our system leverages a sophisticated gensim model to analyze and recommend the top 5 best-matching Coursera courses. 
        Our goal is to streamline the process of finding the right course tailored to each user's unique learning journey.
    ''')
    st.subheader("Techniques")
    st.write('''
        Under the hood, we utilize advanced natural language processing techniques. 
        The text data is meticulously processed using gensim and nltk methods to ensure high-quality recommendations. 
        We remove stopwords, vectorize, and tokenize the English text, refining the data that our gensim model analyzes.
    ''')
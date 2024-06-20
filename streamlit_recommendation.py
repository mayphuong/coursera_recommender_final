import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from dataprep.eda import plot

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

# Custom CSS to inject
custom_css = """
<style>
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    justify-content: space-between;
}
.stTabs [data-baseweb="tab"] {
    flex: 1;
    text-align: center;
    background-color: #F3F0EA;
    color: #0056D2;
    border-radius: 4px 4px 0px 0px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #0056D2;
    color: #F3F0EA;
}

[data-testid="textInputRootElement"]{
    border: none ;
} 


[data-baseweb="base-input"]{
    background-color: rgb(243, 240, 234);
    border: solid ;
} 

[data-baseweb="tab-panel"]{
    color: black;
} 

[data-testid="stHorizontalBlock"] {
  align-items: flex-start;
}

[data-testid="baseButton-secondary"] {
  margin-top: 30px;
  color: white !important;

}

[data-testid="stWidgetLabel"] {
  color: black;
}

code {
    padding: 0.2em 0.4em;
    margin: 0px;
    border-radius: 0.25rem;
    background: rgb(243, 240, 234);
    color: rgb(9, 171, 59);
    font-size: 14px;
    font-weight: bold;
}



</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

st.title("CourseRec for Coursera")
st.markdown("<style>div.stButton > button:first-child { background-color: #0056D2; color: #0056D2; }</style>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["HOME", "FIND A COURSE", "TOP PICKS FOR YOU", "ABOUT US", "ABOUT THIS PROJECT"])

with tab1:  # HOME
    st.subheader("Welcome to CourseRec for Coursera!")
    st.markdown("""
        **CourseRec for Coursera** is your go-to app for discovering the best Coursera courses tailored to your interests and professional goals. 
        
        ##### How It Works:
        - **Find a Course:** Enter a keyword or course name in the search bar to find courses that closely match your query.
        - **Top Picks for You:** Enter your name if you're a Coursera learner. We'll recommend courses based on your learning and review history.
        - Explore course details and select the one that best fits your learning journey.
        
        Start your personalized learning experience with **CourseRec for Coursera** today!
                
    """, unsafe_allow_html=True)

# Feature engineering for the usage of 2 models
## 1. Read data
courses = pd.read_csv("courses.csv", encoding='utf-8')
reviews = pd.read_csv("reviews.csv", encoding='utf-8')

## 2. Data pre-processing

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
courses['Level'].fillna('N/A', inplace=True)
courses['Results'].fillna('N/A', inplace=True)
courses['Unit'].fillna('N/A', inplace=True)
reviews.dropna(subset=['ReviewContent'], inplace=True)

# Strip ' level' from Level
courses['Level'] = courses['Level'].str.replace(' level', '')

# Strip 'By ' from ReviewerName
reviews['ReviewerName'] = reviews['ReviewerName'].str.strip('By ')

with tab2:  # FIND A COURSE (GENSIM)
    ## 1. Engineer features
    # Get the list of string columns
    courses['Course'] = courses[['CourseName', 'Unit', 'Level', 'Results']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    courses['ProcessedCourse'] = courses['Course'].apply(strip_punctuation)\
                                                .apply(strip_multiple_whitespaces)\
                                                .apply(strip_non_alphanum)\
                                                .apply(strip_numeric)\
                                                .apply(lower_to_unicode)\
                                                .apply(remove_stopwords)

    # Tokenize Course
    courses['TokenizedCourse'] = courses['ProcessedCourse'].progress_apply(word_tokenize)

    ## 2. Build Gensim model
    # Create a dictionary representation of the documents
    dictionary = Dictionary(courses['TokenizedCourse'])

    # Create a Corpus: BOW representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in courses['TokenizedCourse']]

    # Use TF-IDF Model to process corpus, obtaining index
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                                num_features = len(dictionary.token2id))

    # Define a function to suggest courses similar to a particular one
    @st.cache_data(ttl="7 days")
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

    ## 3. Save Gensim model
    pkl_gensim = "gensim_model.pkl"
    tfidf.save(pkl_gensim)

    ## 4. Load Gensim model
    gensim_model = models.TfidfModel.load(pkl_gensim)

    ## 5. GUI
    # Create a subheader
    st.subheader("Find a Course")
    sidebar = st.sidebar

    # Create two columns for user input and button
    col1, col2 = st.columns([3, 1])

    # User input in the first column with a unique key
    with col1:
        user_input = st.text_input("What do you want to learn?")

    # Button in the second column
    with col2:
        find_matches = st.button('Find Best Matches', key="find_matches")

    # Initialize session state for selected courses
    if 'gensim_suggested_courses' not in st.session_state:
        st.session_state['gensim_suggested_courses'] = []

    # Define a function to suggest courses similar to user input
    def suggest_courses_gensim(user_input):
        gensim_suggested_courses = similar_gensim(user_input)
        return gensim_suggested_courses

    # Display the suggested courses and their details
    if find_matches:
        gensim_suggested_courses = suggest_courses_gensim(user_input)
        
        # Iterate over the suggested courses and display their details
        for course_name in gensim_suggested_courses:
            # Find the course details
            course_details = courses[courses['CourseName'] == course_name]

            if len(course_details) > 0:
                course_details = course_details.iloc[0]
            
                # Display the course details using st.write or st.table
                st.markdown(f"<h2 style='font-size:125%;'><b>{course_details['CourseName']}</b></h2>", unsafe_allow_html=True)
                st.write("**Description:**", course_details['Results'])
                st.write("**Provider:**", course_details['Unit'])
                st.write("**Average Rating:**", course_details['AvgStar'])
                st.write("**Level:**", course_details['Level'])
                
                # Add a separator for better readability
                st.markdown("---")
            else:
                continue

with tab3:  # TOP PICKS FOR YOU (SVD)
    # Create a subheader
    st.subheader("Top Picks for You")

    ## 1. Remove reviews by Deleted A
    reviews = reviews[reviews.ReviewerName != 'Deleted A']

    ## 2. Build SVD model
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(reviews[['ReviewerName', 'CourseName', 'RatingStar']], reader)

    # Singular value decomposition
    algorithm = SVD()

    # Fit trainset to the SVD model
    trainset = data.build_full_trainset()
    algorithm.fit(trainset)
    
    # 3. Save SVD model
    from surprise import dump
    dump.dump('svd_model', algo=algorithm)

    # 4. Load SVD model
    _, loaded_algorithm = dump.load('svd_model')

    # Define a function to suggest courses to a specific reviewer
    @st.cache_data(ttl="7 days")
    def similar_svd(name, num=5):
        reviewed = reviews[reviews['ReviewerName']==name]['CourseName'].to_list()
        results = reviews[['CourseName']].copy()
        results['EstScore'] = results['CourseName'].apply(lambda x: loaded_algorithm.predict(name, x).est)
        results = results.sort_values(by=['EstScore'], ascending=False).drop_duplicates()
        results = results[~results['CourseName'].isin(reviewed)]['CourseName'].to_list()[:num]
        return results
    
    ## 4. GUI
    # Initialize session state for selected courses
    if 'svd_suggested_courses' not in st.session_state:
        st.session_state['svd_suggested_courses'] = []

    # Define a function to suggest courses similar to user input
    def svd_suggest_courses(user_name_input):
        svd_suggested_courses = similar_svd(user_name_input)
        return svd_suggested_courses

    # Create two columns for user input and button
    col1, col2 = st.columns([3, 1])

    # User input in the first column with a unique key
    with col1:
        user_name_input = st.text_input('Please enter your name:', '')
        st.caption('e.g. Kevin M, Jianfei Z, Aman J, pooja s, Lauren J, etc.')

    # Button in the second column
    with col2:
        login_button = st.button('Login')

    if login_button:
        if user_name_input in reviews['ReviewerName'].values:
            st.success('Successfully logged in!')

            # Update the session state with suggested courses
            st.session_state['svd_suggested_courses'] = svd_suggest_courses(user_name_input)
            # svd_suggested_courses = svd_suggest_courses(user_name_input)

            st.subheader('TOP PICKS FOR YOU')
            # Display the suggested courses and their details
            # if find_matches:
                # svd_suggested_courses = svd_suggest_courses(user_name_input)
            
            # Iterate over the suggested courses and display their details
            for course_name in st.session_state['svd_suggested_courses']:
            # for course_name in svd_suggested_courses:
                # Find the course details
                svd_course_details = courses[courses['CourseName'] == course_name]

                if len(svd_course_details) > 0:
                    svd_course_details = svd_course_details.iloc[0]
                
                    # Display the course details using st.write or st.table
                    st.markdown(f"<h2 style='font-size:125%;'><b>{svd_course_details['CourseName']}</b></h2>", unsafe_allow_html=True)
                    st.write("**Description:**", svd_course_details['Results'])
                    st.write("**Provider:**", svd_course_details['Unit'])
                    st.write("**Average Rating:**", svd_course_details['AvgStar'])
                    st.write("**Level:**", svd_course_details['Level'])
                    
                    # Add a separator for better readability
                    st.markdown("---")
                else:
                    continue
            # for pick in top_picks:
            #     st.write(pick)
        else:
            st.error('No user information found, please try again.')
    
with tab4:
    st.subheader("About Us")

    # Create two columns for the profiles
    col1, col2 = st.columns(2)

    with col1:
        # Profile for Linh N.
        st.markdown("#### Linh N.")
        st.write("Email: linh.n@gmail.com")
        st.write('Phone: +987654321')
        st.write('Role: Data Processing and Modelling')
        st.write('''
            Linh was instrumental in constructing the foundation of our recommendation system. 
            From data collection to preprocessing, Linh ensured the data was clean and structured. 
            Linh's expertise in modeling also contributed significantly to the initial build of our system.
        ''')

    with col2:
        # Profile for Phuong N.
        st.markdown("#### Phuong N.")
        st.write("Email: phuong.n@gmail.com")
        st.write('Phone: +123456789')
        st.write('Role: Model Fine-Tuning')
        st.write('''
            Phuong played a pivotal role in enhancing the performance of our recommendation system. 
            With a keen eye for detail, Phuong meticulously fine-tuned the model parameters to improve accuracy and ensure the most relevant course suggestions.
        ''')

    # Add a section for shared responsibilities
    st.write('''
        Both Phuong N. and Linh N. brought their unique strengths to the table in a collaborative effort on the app design. 
        Their combined insights have been invaluable in creating an intuitive and user-friendly interface that enhances the overall user experience.
    ''')


with tab5:
    st.subheader("About This Project")

    # Introduction
    st.markdown("#### Introduction")
    st.markdown("Our project aims to develop a recommendation system for Coursera, a prominent technological educational platform. This system is designed to enhance the user experience by guiding learners towards relevant learning courses.")

    # Recommendation Modes
    st.markdown("#### Recommendation Modes")
    st.markdown("The system operates in two primary modes:")
    st.markdown("- **Content-based Filtering:** Users can find courses by entering search queries related to their interests, skill sets, or desired technologies.")
    st.markdown("- **Collaborative Filtering:** Leveraging profiles of similar learners, the system recommends courses based on their preferences and learning history on Coursera.")

    # Machine Learning Models
    st.markdown("#### Machine Learning Models")
    st.markdown("Recommendations are generated using two machine learning models:")
    st.markdown("- **GenSim with TFIDF:** This model processes textual data to recommend courses based on keyword relevance.")
    st.markdown("- **SVD (Singular Value Decomposition):** This model analyzes user-course interactions to provide personalized recommendations.")

    # Data Sources
    st.markdown("#### Data Sources")
    st.markdown("The models are trained on a dataset extracted from Coursera, focusing specifically on courses within data science and machine learning. The dataset includes comprehensive information about courses and user reviews, ensuring accurate and effective recommendations.")

    # Project Goals
    st.markdown("#### Project Goals")
    st.markdown("By providing personalized course suggestions, our project aims to encourage learners to explore a diverse range of courses and foster sustained engagement with Coursera.")
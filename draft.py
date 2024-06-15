import numpy as np
import pandas as pd
import pickle
import streamlit as st

# from dataprep.eda import plot
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import re

from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import *
from gensim.utils import *
from numpy import dot
from numpy.linalg import norm

from surprise import Reader, Dataset, SVD
from surprise.model_selection.validation import cross_validate
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import warnings


# 1. Read data
courses = pd.read_csv("courses.csv", encoding='utf-8')
reviews = pd.read_csv("reviews.csv", encoding='utf-8')

#--------------
# GUI
st.title("Recommenation System Project")
st.write("Coursera Courses")

# Upload file
# uploaded_file = st.file_uploader("Choose a file", type=['csv'])
# if uploaded_file is not None:
#     courses = pd.read_csv(uploaded_file, encoding='latin-1')
#     courses.to_csv("spam_new.csv", index = False)

# 2. Data pre-processing

# Drop CourseID as it's identical to CourseName
courses = courses.drop('CourseID', axis=1)

# Strip "By" from ReviewerName
reviews['ReviewerName'] = reviews['ReviewerName'].str.strip('By ')
reviews.head()

# Engineer features
courses['Results'] = courses['Results'].fillna('')
courses['description'] = courses[['CourseName', 'Results']].apply(lambda x: ' '.join(x), axis=1)
content = courses[['CourseName','description']].copy()

# Remove reviews by Deleted A
reviews = reviews[reviews.ReviewerName != 'Deleted A']

# Preprocess with Gensim
content['description'] = content['description'].astype(str)
content['description'] = content['description'].apply(strip_punctuation).apply(strip_multiple_whitespaces).apply(strip_non_alphanum).apply(strip_numeric).apply(remove_stopwords).apply(simple_preprocess)

# 3. Build model

# Obtain the number of features based on dictionary: Use corpora.Dictionary
dictionary = corpora.Dictionary(content['description'])

# Obtain corpus based on dictionary (dense matrix)
corpus = [dictionary.doc2bow(text) for text in content['description']]

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
  # Print results
  print(f"Similar courses to '{course}':\n")
  for result in results:
    print(result)

#4. Evaluate model

# Test the model with random courses
similar_gensim(content.iloc[18]['CourseName'])
similar_gensim(content.iloc[40]['CourseName'])
similar_gensim(content.iloc[118]['CourseName'], 7)

# score_train = model.score(X_train,y_train)
# score_test = model.score(X_test,y_test)
# acc = accuracy_score(y_test,y_pred)
# cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# cr = classification_report(y_test, y_pred)

# y_prob = model.predict_proba(X_test)
# roc = roc_auc_score(y_test, y_prob[:, 1])

#5. Save models
# Save Gensim model
pkl_filename = "gensim_model.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(similar_gensim, file)
pickle.dump(similar_gensim, open(pkl_filename, 'wb'))
  
# # luu model CountVectorizer (count)
# pkl_count = "count_model.pkl"  
# with open(pkl_count, 'wb') as file:  
#     pickle.dump(count, file)


#6. Load models 
# Đọc model
# import pickle
with open(pkl_filename, 'rb') as file:  
    gensim_model = pickle.load(file)

pickle.__file__
# # doc model count len
# with open(pkl_count, 'rb') as file:  
#     count_model = pickle.load(file)

# # GUI
# menu = ["Business Objective", "Build Project", "New Prediction"]
# choice = st.sidebar.selectbox('Menu', menu)
# if choice == 'Business Objective':    
#     st.subheader("Business Objective")
#     st.write("""
#     ###### Classifying spam and ham messages is one of the most common natural language processing tasks for emails and chat engines. With the advancements in machine learning and natural language processing techniques, it is now possible to separate spam messages from ham messages with a high degree of accuracy.
#     """)  
#     st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for ham and spam message classification.""")
#     st.image("ham_spam.jpg")

# elif choice == 'Build Project':
#     st.subheader("Build Project")
#     st.write("##### 1. Some data")
#     st.dataframe(courses[['v2', 'v1']].head(3))
#     st.dataframe(courses[['v2', 'v1']].tail(3))  
#     st.write("##### 2. Visualize Ham and Spam")
#     fig1 = sns.countplot(data=courses[['v1']], x='v1')    
#     st.pyplot(fig1.figure)

#     st.write("##### 3. Build model...")
#     st.write("##### 4. Evaluation")
#     st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
#     st.code("Accuracy:"+str(round(acc,2)))
#     st.write("###### Confusion matrix:")
#     st.code(cm)
#     st.write("###### Classification report:")
#     st.code(cr)
#     st.code("Roc AUC score:" + str(round(roc,2)))

#     # calculate roc curve
#     st.write("###### ROC curve")
#     fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
#     fig, ax = plt.subplots()       
#     ax.plot([0, 1], [0, 1], linestyle='--')
#     ax.plot(fpr, tpr, marker='.')
#     st.pyplot(fig)

#     st.write("##### 5. Summary: This model is good enough for Ham vs Spam classification.")

# elif choice == 'New Prediction':
#     st.subheader("Select data")
#     flag = False
#     lines = None
#     type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
#     if type=="Upload":
#         # Upload file
#         uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
#         if uploaded_file_1 is not None:
#             lines = pd.read_csv(uploaded_file_1, header=None)
#             st.dataframe(lines)
#             # st.write(lines.columns)
#             lines = lines[0]     
#             flag = True       
#     if type=="Input":        
#         email = st.text_area(label="Input your content:")
#         if email!="":
#             lines = np.array([email])
#             flag = True
    
#     if flag:
#         st.write("Content:")
#         if len(lines)>0:
#             st.code(lines)        
#             x_new = count_model.transform(lines)        
#             y_pred_new = ham_spam_model.predict(x_new)       
#             st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new))
    


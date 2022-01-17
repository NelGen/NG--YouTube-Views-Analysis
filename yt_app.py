import streamlit as st
from Functions import *

# Set Page Title and Layout
st.set_page_config(page_title = "YT View Predictor",layout="wide")

# Page Header
st.write("""
# YouTube Views Predictor
""")

st.caption("Specifically for use with LowkoTV")

# Creates text field for video title input
form = st.form(key='my-form')
user_input = form.text_input('Enter video title and submit to predict views:')
submit = form.form_submit_button('Submit')

# Takes the text input and predicts number of views and creates bargraph of average views by word
if submit:
    df,not_found = preprocessor(user_input)
    prediction = "Predicted # of Views: " + str(view_predict(user_input))
    st.header(prediction)
    
    if not_found:
        joined = ", ".join(not_found)
        st.write("Words not seen before or filtered out:",joined, "** Try writing the words in all caps")
    
    st.write("## Word Visualization")
    col1, col2 = st.columns(2)
    col1.pyplot(visualizer(user_input))
    

# Pulls pre generated graphs for review
st.subheader("Other Select Visualizations From YouTube Analysis")

col3, col4 = st.columns(2)

# Generic stat visualizations
col3.subheader("Some Generic Stats")
goption = col3.selectbox("Select from the available Visualizations",
                      ("None",
                       "Views By Video Length",
                       "Average Views Per Game",
                       "Views By Video Length Per Game",
                       "Average Views By Certain Key Words"))
# Pulls the related image
if goption != "None":
    gimage = r"App Images/" + goption + ".png"
    col3.image(gimage)

# StarCraft 2 related visualizations
col4.subheader("SC2 Related Stats")
sc2option = col4.selectbox("Select from the available Visualizations",
                           ("None",
                            "Average Views By SC2 Matchups",
                            "Average Views By SC2 Player",
                            "Average Views By Player's SC2 Race",
                            "Average Views By SC2 Player's Country",
                            "Average Views By SC2 Race and Video Length"))
if sc2option != "None":
    sc2image = r"App Images/" + sc2option + ".png"
    col4.image(sc2image)
    
                            







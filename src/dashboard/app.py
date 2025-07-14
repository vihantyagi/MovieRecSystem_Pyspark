# src/dashboard/app_simple.py
import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(page_title="Movie Recommendations", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")

# API URL - using localhost since we confirmed it works
API_URL = "http://localhost:8000"

# Sidebar
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value=API_URL)

# Test API connection
try:
    response = requests.get(f"{api_url}/")
    health = response.json()
    if health["status"] == "healthy":
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except Exception as e:
    st.sidebar.error(f"❌ Connection Error: {str(e)}")

# Main content
tab1, tab2, tab3 = st.tabs(["Get Recommendations", "Model Info", "Test"])

with tab1:
    st.header("Get Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        user_id = st.number_input("User ID", min_value=0, value=1)
        num_recs = st.slider("Number of recommendations", 5, 20, 10)
        
        if st.button("Get Recommendations"):
            try:
                response = requests.post(
                    f"{api_url}/recommend",
                    json={"user_id": user_id, "num_recommendations": num_recs}
                )
                if response.status_code == 200:
                    st.session_state['recs'] = response.json()
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # with col2:
    #     if 'recs' in st.session_state:
    #         st.subheader(f"Recommendations for User {st.session_state['recs']['user_id']}")
    #         for rec in st.session_state['recs']['recommendations']:
    #             st.write(f"**Rank {rec['rank']}**: Movie {rec['movie_id']} - Rating: {rec['predicted_rating']:.2f}")

    with col2:
        if 'recs' in st.session_state:
            st.subheader(f"Recommendations for User {st.session_state['recs']['user_id']}")
            for rec in st.session_state['recs']['recommendations']:
                # Display with title if available
                if 'title' in rec and rec['title']:
                    st.write(f"**Rank {rec['rank']}**: {rec['title']}")
                    st.write(f"   Predicted Rating: {'⭐' * int(rec['predicted_rating'])} ({rec['predicted_rating']:.2f})")
                else:
                    st.write(f"**Rank {rec['rank']}**: Movie {rec['movie_id']} - Rating: {rec['predicted_rating']:.2f}")
                st.write("---")
with tab2:
    st.header("Model Information")
    try:
        response = requests.get(f"{api_url}/model/info")
        if response.status_code == 200:
            info = response.json()
            st.json(info)
    except Exception as e:
        st.error(f"Error: {str(e)}")

with tab3:
    st.header("API Test")
    if st.button("Get Random Users"):
        try:
            response = requests.get(f"{api_url}/users/random?n=5")
            if response.status_code == 200:
                st.json(response.json())
        except Exception as e:
            st.error(f"Error: {str(e)}")

import streamlit as st
from PIL import Image


# Customize the sidebar
markdown = """
Web App URL: <https://template.streamlit.app>
GitHub Repository: <https://github.com/giswqs/streamlit-multipage-template>
"""


# Customize page title
st.title("Web Application to make analysis and predict financial data")
st.subheader(
    """
    This app contains multiple models to analyse and predict stock market data.
    """
)
image1 = Image.open('/Users/aameerkhan/Desktop/Streamlit-multipage/assests/stockimg.png')
st.image(image1,caption="image displaying an analysis of amazon stock dataset.",width=300)



st.subheader("This app also has the feature to use Twitter analysis and also use OpenAi's ChatGpt feature to search and gather insights about the current trend")

col1, col2 = st.columns(2)

with col1:
    chatgptimg = Image.open('/Users/aameerkhan/Desktop/Streamlit-multipage/assests/chatgpt.webp')
    st.image(chatgptimg,caption="ChatGpt created by OpenAi")
with col2:
    twitterimg = Image.open('/Users/aameerkhan/Desktop/Streamlit-multipage/assests/twitterimg.png')
    st.image(twitterimg,caption="Twitter Logo",width=320)

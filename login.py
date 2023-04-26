import streamlit as st

# Define actual email and password
actual_email = "aameerk917@gmail.com"
actual_password = "password"
st.header("LOGIN DETAILS")
# Create an empty container
placeholder = st.empty()

with placeholder.form("login"):
    st.markdown("#### Enter your credentials")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

    # If the form is submitted and the email and password are correct,
    # display a success message in Streamlit app and redirect to Flask app
    if submit and email == actual_email and password == actual_password:
        placeholder.empty()
        st.success("Login successful")
        st.experimental_set_query_params(username=email)
        iframe = f'<iframe src="http://localhost:5000/success" width="700" height="400"></iframe>'
        st.components.v1.html(iframe, width=800, height=500)

    # If the form is submitted and the email or password is incorrect,
    # display an error message in Streamlit app
    elif submit and (email != actual_email or password != actual_password):
        st.error("Login failed")
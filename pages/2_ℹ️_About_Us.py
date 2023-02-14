import streamlit as st
from PIL import Image

def run_ui():
    
    '''
    Renders the Streamlit Page UI
    '''
    
    # setting streamlit page configuration
    st.set_page_config(
        layout="wide",
        page_title="About Us",
        page_icon="ℹ️")
    
    st.sidebar.write("Image content moderation is crucial in ensuring the safety and well-being of individuals who use digital platforms. With the widespread use of the internet, it's become easier to access and share images, both good and bad.")
    st.sidebar.write("This ease of access to visual content has also led to an increase in the spread of inappropriate, violent, and pornographic images. These types of images can be harmful and traumatizing for those who come across them, particularly children and young people.")
    st.sidebar.write("By filtering out inappropriate, violent, and pornographic images, digital platforms can create a safer and more inclusive environment for all users. Moderating these types of images can also help prevent the spread of harmful beliefs and legal consequences for platform operators.")

    st.title("Content Moderation for Social Media")
    st.markdown("""---""")
    st.subheader("About Us")

    ## Displays the team members
    row_1_col1, row_1_col2, row_1_col3, row_1_col4, row_1_col5 = st.columns(5)
    with row_1_col2:
        image = Image.open('./assets/andrew.jpg')
        st.image(image, caption="Andrew Bonafede")
    with row_1_col3:
        image = Image.open('./assets/shrey.jpg')
        st.image(image, caption="Shrey Gupta")
    with row_1_col4:
        image = Image.open('./assets/shuai.jpg')
        st.image(image, caption="Shuai Zheng")

    row_2_col2, row_2_col3, row_2_col4 = st.columns((1,5,1))
    # with row_2_col2:
    #     st.write("Andrew")

    with row_2_col3:
        st.markdown("<h5 style='text-align: justify;'>We are a team of 3 students from Duke University, studying Artificial Intelligence. We are passionate about artificial intelligence and machine learning, and we are excited to be working on this project as a part of one of our core courses - AIPI540: Deep Learning Applications!</h5>", unsafe_allow_html=True)


    # with row_2_col4:
    #     st.write("Shuai")

if __name__ == "__main__":
    run_ui()
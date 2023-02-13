import streamlit as st
from PIL import Image
from clf import predict

def run_ui():
    # setting page layout to wide
    st.set_page_config(
            page_title="Content Moderation for Social Media",
            page_icon="🌐",
            layout="wide")

    # setting page title
    st.title("Content Moderation for Social Media")
    st.markdown("""---""")

    # adding text to sidebar
    st.sidebar.write("Image content moderation is crucial in ensuring the safety and well-being of individuals who use digital platforms. With the widespread use of the internet, it's become easier to access and share images, both good and bad.")
    st.sidebar.write("This ease of access to visual content has also led to an increase in the spread of inappropriate, violent, and pornographic images. These types of images can be harmful and traumatizing for those who come across them, particularly children and young people.")
    st.sidebar.write("By filtering out inappropriate, violent, and pornographic images, digital platforms can create a safer and more inclusive environment for all users. Moderating these types of images can also help prevent the spread of harmful beliefs and legal consequences for platform operators.")

    image_upload = st.file_uploader("Upload an image", type="jpg")

    col1, col2, col3 = st.columns((3,3,3))

    if image_upload is not None:
        image = Image.open(image_upload)
        with col2:
            st.image(image, caption='Uploaded Image.', width=500)
            st.write("")

        labels = predict(image_upload)

        col4, col5, col6, col7, col8, col9 = st.columns((3,3,3,3,3,3))
        with col6:
            st.metric(label=labels[0][0].upper() + " " + "Content Score", value=format(labels[0][1]*100, "0.01f")+"%")

        with col7:
            st.metric(label=labels[1][0].upper() + " " + "Content Score", value=format(labels[1][1]*100, "0.01f")+"%")

        with col8:
            st.metric(label=labels[2][0].upper() + " " + "Content Score", value=format(labels[2][1]*100, "0.01f")+"%")

if __name__ == "__main__":
    run_ui()
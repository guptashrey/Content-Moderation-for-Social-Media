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
    st.sidebar.write("Influenza is the most common viral infection with virus strains mutating every season, effectively changing the severity of the infection. In rare but possible cases this could be deadly especially for the high-risk groups. Medical service providers can face situations where they have deployed more resources than necessary or they are entirely overwhelmed.")
    st.sidebar.write("This will serve as a tool for the decision makers to promote relevant measures to curb infections in different regions through vaccination drives, masking advisories to the public, better crowd control, better resource management in terms of hospitalizations and better gauge of the possibility of the flu turning into an epidemic or a pandemic.")

    image_upload = st.file_uploader("Upload an image", type="jpg")

    if image_upload is not None:
        image = Image.open(image_upload)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
        labels = predict(image_upload)

        # print out the top 5 prediction labels with scores
        for i in labels:
            st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])

if __name__ == "__main__":
    run_ui()
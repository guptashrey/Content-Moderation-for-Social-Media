# Import the required functions from other modules
from scripts.build_features import optimize_image_path
from scripts.make_dataset import download_image_path
from scripts.model import train_predict_dl_model, train_predict_non_dl_model

# Define the main function to run the pipeline
def run():
    # Download the neutral images and save them in the "./data/processed/neutral" folder
    download_image_path("./data/processed/neutral", "./failed_downloads.txt", "./data/raw/urls_neutral.txt")
    # Download the porn images and save them in the "./data/processed/porn" folder
    download_image_path("./data/processed/porn", "./failed_downloads.txt", "./data/raw/urls_porn.txt")
    # Optimize the porn images and save them in the "./data/outputs/porn" folder
    optimize_image_path("./data/processed/porn", "./data/outputs/porn")
    # Optimize the neutral images and save them in the "./data/outputs/neutral" folder
    optimize_image_path("./data/processed/neutral", "./data/outputs/neutral")
    # Train and predict using a deep learning model, and save the model to "./models/best_model.pt"
    train_predict_dl_model(image_path='./data/outputs', model_save_path='./models/best_model.pt')
    # Train and predict using a non-deep learning model, and save the model to "./models/sklearn_non_dl_best_model.pkl"
    train_predict_non_dl_model(image_path='./data/outputs', model_save_path='./models/sklearn_non_dl_best_model.pkl')

# Call the main function when the script is run as the main module
if __name__ == '__main__':
    run()

from scripts.build_features import optimize_image_path
from scripts.make_dataset import download_image_path
from scripts.model import train_predict_dl_model, train_predict_non_dl_model

if __name__ == '__main__':
    download_image_path("./data/processed/neutral", "./failed_downloads.txt", "./data/raw/urls_neutral.txt")
    download_image_path("./data/processed/porn", "./failed_downloads.txt", "./data/raw/urls_porn.txt")
    optimize_image_path("./data/processed/porn", "./data/outputs/porn")
    optimize_image_path("./data/processed/neutral", "./data/outputs/neutral")
    train_predict_dl_model(image_path='./data/outputs', model_save_path='./models/best_model.pt')
    train_predict_non_dl_model(image_path='./data/outputs', model_save_path='./models/sklearn_non_dl_best_model.pkl')

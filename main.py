from scripts.model import train_predict_dl_model,train_predict_non_dl_model



if __name__ == '__main__':
    train_predict_dl_model(image_path='./images_small', model_save_path='./models/best_model.pt')
    train_predict_non_dl_model(image_path='./images_small', model_save_path='./models/sklearn_non_dl_best_model.pkl')

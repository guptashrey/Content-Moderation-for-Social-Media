from multiprocessing import Pool
from PIL import Image
import os

def optimize_image(filename):
    folder = './violence_images'
    opt_folder = './violence'
    image_path = os.path.join(folder, filename)
    opt_image_path = os.path.join(opt_folder, filename)
    try:
        image = Image.open(image_path)
        im2 = image.convert('RGB')

        print(f"optimize image '{image_path}'")
        image.save(opt_image_path, optimize=True, quality=80)
        os.remove(image_path)
    except Exception as e:
        print(f"Failed to download image '{image_path}': {e}")
        os.remove(image_path)

if __name__ == '__main__':
    folder = './violence_images'
    opt_folder = './violence'
    file_names = os.listdir(folder)
    with Pool(8) as p:
        print(p.map(optimize_image, file_names))
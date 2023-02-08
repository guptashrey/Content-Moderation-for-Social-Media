from PIL import Image
import os

def optimize_image(image_path):
    image = Image.open(image_path)
    image.save(image_path, optimize=True, quality=95)

folder = '/Users/shuai/PycharmProjects/Content-Moderation-for-Social-Media/images/neutral'
for filename in os.listdir(folder):
    image_path = os.path.join(folder, filename)
    optimize_image(image_path)
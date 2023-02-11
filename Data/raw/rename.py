import os

def rename_image(image_path):
    head, tail = os.path.split(image_path)
    name, ext = os.path.splitext(tail)
    new_name = name.split("?")[0] + ext
    new_path = os.path.join(head, new_name)
    os.rename(image_path, new_path)
    os.remove(image_path)

folder = '/Users/shuai/PycharmProjects/Content-Moderation-for-Social-Media/images/neutral'
for filename in os.listdir(folder):
    image_path = os.path.join(folder, filename)
    rename_image(image_path)
from PIL import Image
import os


def optimize_image(image_path, output_path):
    try:
        image = Image.open(image_path)
        im2 = image.convert('RGB')

        print(f"optimize image '{image_path}'")
        image.save(output_path, optimize=True, quality=80)
    except Exception as e:
        print(f"Failed to optimize image '{image_path}': {e}")
        # os.remove(image_path)


def optimize_image_path(folder, output_path):
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        optimize_image(image_path, output_path)

#
# if __name__ == '__main__':
#     folder = '/Users/shuai/Downloads/nsfw'
#     for filename in os.listdir(folder):
#         image_path = os.path.join(folder, filename)
#         optimize_image(image_path)

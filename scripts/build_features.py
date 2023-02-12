from PIL import Image
import os

# open the image in the path and optimize it with quality 80 and save the optimized image in the output path
def optimize_image(image_path, output_path):
    try:
        image = Image.open(image_path)
        im2 = image.convert('RGB')

        print(f"optimize image '{image_path}'")
        image.save(output_path, optimize=True, quality=80)
    except Exception as e:
        print(f"Failed to optimize image '{image_path}': {e}")
        # os.remove(image_path)


# load all images in the path and optimize it with quality 80 and save the optimized image in the output path
def optimize_image_path(folder, output_folder):
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        output_path = os.path.join(output_folder, filename)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(image_path,output_path)
        optimize_image(image_path,output_path)

#
# if __name__ == '__main__':
#     folder = '/Users/shuai/Downloads/nsfw'
#     for filename in os.listdir(folder):
#         image_path = os.path.join(folder, filename)
#         optimize_image(image_path)

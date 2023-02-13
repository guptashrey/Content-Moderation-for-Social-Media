# library imports
from PIL import Image
import os

def optimize_image(image_path, output_path):
    """
    Optimize a single image
    """

    try:
        # open the image using the Image.open method
        image = Image.open(image_path)

        # convert the image to RGB format
        im2 = image.convert('RGB')

        # print a message indicating that the image is being optimized
        print(f"optimize image '{image_path}'")

        # save the optimized image with a quality of 80
        # the optimize argument is set to True to optimize the file format
        image.save(output_path, optimize=True, quality=80)
    
    except Exception as e:
        # print a message if there was a failure to optimize the image
        print(f"Failed to optimize image '{image_path}': {e}")
        # Uncomment the next line to remove the image if optimization fails
        # os.remove(image_path)

def optimize_image_path(folder, output_folder):
    """
    Optimize all images in a given folder
    """
    # Loop through all files in the folder
    for filename in os.listdir(folder):
        # construct the path to the current file
        image_path = os.path.join(folder, filename)
        # construct the path to the output file
        output_path = os.path.join(output_folder, filename)
        
        # create the output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # print the input and output paths for each file
        print(image_path,output_path)
        # call the optimize_image function to optimize the current file
        optimize_image(image_path,output_path)

if __name__ == '__main__':
    folder = '../data/images/nsfw'
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        optimize_image(image_path)
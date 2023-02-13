# Import the Image module from the PIL library
from PIL import Image
import os

# Function to optimize a single image
def optimize_image(image_path, output_path):
    try:
        # Open the image using the Image.open method
        image = Image.open(image_path)

        # Convert the image to RGB format
        im2 = image.convert('RGB')

        # Print a message indicating that the image is being optimized
        print(f"optimize image '{image_path}'")

        # Save the optimized image with a quality of 80
        # The optimize argument is set to True to optimize the file format
        image.save(output_path, optimize=True, quality=80)
    except Exception as e:
        # Print a message if there was a failure to optimize the image
        print(f"Failed to optimize image '{image_path}': {e}")
        # Uncomment the next line to remove the image if optimization fails
        # os.remove(image_path)

# Function to optimize all images in a given folder
def optimize_image_path(folder, output_folder):
    # Loop through all files in the folder
    for filename in os.listdir(folder):
        # Construct the path to the current file
        image_path = os.path.join(folder, filename)
        # Construct the path to the output file
        output_path = os.path.join(output_folder, filename)
        # Create the output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Print the input and output paths for each file
        print(image_path,output_path)
        # Call the optimize_image function to optimize the current file
        optimize_image(image_path,output_path)

# Uncomment the next block of code to run the optimization process on all files in a folder
# if __name__ == '__main__':
#     folder = '/Users/shuai/Downloads/nsfw'
#     for filename in os.listdir(folder):
#         image_path = os.path.join(folder, filename)
#         optimize_image(image_path)

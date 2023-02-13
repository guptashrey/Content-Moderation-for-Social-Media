# Import the required libraries
import requests
import os
from PIL import Image

# Function to download a single image
def download_image(url, name, path, path_failurls):
    # Remove any query parameters from the name of the image
    name = name.split("?")[0]
    # Construct the path to the output file
    image_path = os.path.join(path, name)
    # Check if the image already exists at the specified path
    if os.path.exists(image_path):
        # If the image exists, print a message and return
        print(f"Image '{name}' already exists, skipping.")
        return

    # Check if the file has a supported image format
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        # .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp
        try:
            # Send a GET request to the URL to download the image
            response = requests.get(url, timeout=20)
            # Construct the path to the output file
            file_path = image_path
            # Write the image content to the output file
            with open(file_path, "wb") as f:
                f.write(response.content)

            # Print a message indicating that the image was successfully downloaded
            print(f"Image '{name}' downloaded at path '{file_path}'.")

        except Exception as e:
            # If there was an error, print a message and add the URL to the list of failed downloads
            print(f"Failed to download image '{name}': {e}")
            with open(path_failurls, "a") as f:
                f.write(f"{url}\n")

# Function to download all images in a given list
def download_image_path(path, path_failurls, path_image_url):
    # Load the list of failed downloads
    failed_downloads = set()
    if os.path.exists("%s" % path_failurls):
        with open(path_failurls) as f:
            for line in f:
                failed_downloads.add(line.strip())
    # Create the output folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    # Read the list of image URLs
    with open("%s" % path_image_url) as f:
        for url in f:
            # Strip the newline character from the URL
            url = url.strip()
            # Check if the URL is in the list of failed downloads
            if url in failed_downloads:
                # If the URL is in the list of failed downloads, print a message and skip it
                print(f"Image with URL '{url}' has failed to download before, skipping.")
                continue
            # Extract the name of the image from the URL
            name = url.split("/")[-1]
            # Call the download_image function to download the current image
            download_image(url, name, path, path_failurls)


# if __name__ == '__main__':
#     path = "../../images/normal"
#     path_failurls = "./failed_downloads.txt"
#     path_image_url = "./urls_neutral.txt"
#
#     download_image_path(path, path_failurls, path_image_url)

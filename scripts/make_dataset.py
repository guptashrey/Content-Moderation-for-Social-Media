# library imports
import requests
import os
from PIL import Image

def download_image(url, name, path, path_failurls):
    """
    Download an image from a given URL and save it to a given path.
    """
    # remove any query parameters from the name of the image
    name = name.split("?")[0]

    # construct the path to the output file
    image_path = os.path.join(path, name)

    # check if the image already exists at the specified path
    if os.path.exists(image_path):
        # if the image exists, print a message and return
        print(f"Image '{name}' already exists, skipping.")
        return

    # check if the file has a supported image format
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        # if the file has a supported image format, download the image
        try:
            # send a GET request to the URL to download the image
            response = requests.get(url, timeout=20)
            # construct the path to the output file
            file_path = image_path
            # write the image content to the output file
            with open(file_path, "wb") as f:
                f.write(response.content)

            # print a message indicating that the image was successfully downloaded
            print(f"Image '{name}' downloaded at path '{file_path}'.")

        except Exception as e:
            # if there was an error, print a message and add the URL to the list of failed downloads
            print(f"Failed to download image '{name}': {e}")
            with open(path_failurls, "a") as f:
                f.write(f"{url}\n")

def download_image_path(path, path_failurls, path_image_url):
    """
    Download all images in a given list
    """
    # empty set for failed downloads
    failed_downloads = set()
    if os.path.exists("%s" % path_failurls):
        with open(path_failurls) as f:
            for line in f:
                failed_downloads.add(line.strip())

    # create the output folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # read the list of image URLs
    with open("%s" % path_image_url) as f:
        for url in f:
            # strip the newline character from the URL
            url = url.strip()

            # check if the URL is in the list of failed downloads
            if url in failed_downloads:
                # if the URL is in the list of failed downloads, print a message and skip it
                print(f"Image with URL '{url}' has failed to download before, skipping.")
                continue

            # extract the name of the image from the URL
            name = url.split("/")[-1]
            # call the download_image function to download the current image
            download_image(url, name, path, path_failurls)

if __name__ == '__main__':
    path = "../data/images/neutral"
    path_failurls = "./failed_downloads.txt"
    path_image_url = "../data/source_urls/neutral.txt"

    download_image_path(path, path_failurls, path_image_url)
import requests
import os
from PIL import Image



def download_image(url, name, path,path_failurls):
    name=name.split("?")[0]
    image_path = os.path.join(path, name)
    if os.path.exists(image_path):
        print(f"Image '{name}' already exists, skipping.")
        return

    try:
        response = requests.get(url, timeout=20)
        file_path = image_path
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Image '{name}' downloaded at path '{file_path}'.")
        image = Image.open(image_path)
        im2 = image.convert('RGB')

        print(f"optimize image '{image_path}'")
        image.save(image_path, optimize=True, quality=80)
    except Exception as e:
        print(f"Failed to download image '{name}': {e}")
        with open(path_failurls, "a") as f:
            f.write(f"{url}\n")


def download_image_path(path, path_failurls, path_image_url):
    failed_downloads = set()
    if os.path.exists("%s" % path_failurls):
        with open(path_failurls) as f:
            for line in f:
                failed_downloads.add(line.strip())
    if not os.path.exists(path):
        os.makedirs(path)
    with open("%s" % path_image_url) as f:
        for url in f:
            url = url.strip()
            if url in failed_downloads:
                print(f"Image with URL '{url}' has failed to download before, skipping.")
                continue
            name = url.split("/")[-1]
            download_image(url, name, path, path_failurls)


# if __name__ == '__main__':
#     path = "../../images/normal"
#     path_failurls = "./failed_downloads.txt"
#     path_image_url = "./urls_neutral.txt"
#
#     download_image_path(path, path_failurls, path_image_url)

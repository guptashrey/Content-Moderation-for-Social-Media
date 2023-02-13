import requests
import os



def download_image(url, name, path,path_failurls):
    name=name.split("?")[0]
    if os.path.exists(os.path.join(path, name)):
        print(f"Image '{name}' already exists, skipping.")
        return

    try:
        response = requests.get(url, timeout=20)
        file_path = os.path.join(path, name)
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Image '{name}' downloaded at path '{file_path}'.")
    except Exception as e:
        print(f"Failed to download image '{name}': {e}")
        with open(path_failurls, "a") as f:
            f.write(f"{url}\n")


if __name__ == '__main__':
    path = "./images/neutral"
    path_failurls = "./failed_downloads.txt"
    path_image_url = "./Data/urls_neutral.txt"

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
            download_image(url, name, path,path_failurls)

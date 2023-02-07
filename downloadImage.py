import requests
import os


def download_image(url, name, path):
    if os.path.exists(os.path.join(path, name)):
        print(f"Image '{name}' already exists, skipping.")
        return

    try:
        response = requests.get(url, timeout=10)
        file_path = os.path.join(path, name)
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Image '{name}' downloaded at path '{file_path}'.")
    except Exception as e:
        print(f"Failed to download image '{name}': {e}")
        with open("failed_downloads.txt", "a") as f:
            f.write(f"{url}\n")


if __name__ == '__main__':
    path = "./images/neutral"
    failed_downloads = set()

    if os.path.exists("failed_downloads.txt"):
        with open("failed_downloads.txt") as f:
            for line in f:
                failed_downloads.add(line.strip())

    if not os.path.exists(path):
        os.makedirs(path)

    with open("./Data/urls_neutral.txt") as f:
        for url in f:
            url = url.strip()
            if url in failed_downloads:
                print(f"Image with URL '{url}' has failed to download before, skipping.")
                continue
            name = url.split("/")[-1]
            download_image(url, name, path)

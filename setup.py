"""Python setup.py for Content-Moderation-for-Social-Media package"""
import io
import os
from setuptools import find_packages, setup



def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("Content-Moderation-for-Social-Media", "1.0.0")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
            os.path.join(os.path.dirname(__file__), *paths),
            encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


with open("requirements.txt") as f:
    required = f.read().splitlines()
setup(
    name="Content_Moderation_for_Social_Media",
    version='1.0.0',
    description="Content Moderation for Social Media",
    url="https://github.com/guptashrey/Content-Moderation-for-Social-Media",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Shuai, Shrey, Andrew",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=required
)

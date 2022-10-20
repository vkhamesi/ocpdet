import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ocpdet",
    version="0.0.1",
    author="Victor Khamesi",
    author_email="victorkhamesi@hotmail.fr",
    description="A Python library for online changepoint detection, implementing well-known and recent algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vkhamesi/ocpdet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
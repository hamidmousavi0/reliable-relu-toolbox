import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "rrelu",
    version = "0.0.9",
    author = "Seyedhamidreza Mousavi",
    author_email = "seyedhamidreza.mousavi@mdu.se",
    description = "generate reliable relu activation function for DNNs",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/hamidmousavi0/reliable-relu-toolbox",
    project_urls = {
        "Bug Tracker": "https://github.com/hamidmousavi0/reliable-relu-toolbox/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)

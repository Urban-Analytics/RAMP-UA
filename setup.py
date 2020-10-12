import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RAMP-UA",
    version="1.0.0dev",
    author="RAMP UA Team",
    author_email="N.S.Malleson@leeds.ac.uk",
    description="The RAMP Urban Analytics package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Urban-Analytics/RAMP-UA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7"
    ],
    python_requires='>=3.6',
)
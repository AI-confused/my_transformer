import setuptools

with open("README.org", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name='my_transformer',
    version='0.0.1',
    description='my specificaion transformer',
    long_description=long_description,
    long_description_content_type="text/org",
    url='https://github.com/AI-confused/my_transformer',
    author='AI-confused',
    author_email='liyunliang@zju.edu.cn',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

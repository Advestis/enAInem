from setuptools import setup, find_packages

setup(
    name='enainem',
    version='1.3.3',
    description='Robust Non-Negative Tensor Factorization with Integrated Sources and Random Completions',
    author='Paul Fogel, Christophe Geissler, George Luta',
    author_email='paul.fogel@forvismazars.com',
    url='https://github.com/Advestis/enAInem',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "numpy==2.0.2",
        "scipy==1.13.1",
        "pandas==2.2.3",
        "scikit-learn==1.6.0",
        "dask==2024.8.0"
    ],
    extras_require={
        "dev": [
            "matplotlib==3.9.4",
            "ipython==8.31.0"
        ]
    },
    python_requires='>=3.8',
)
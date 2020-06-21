from setuptools import setup, find_packages

setup(
    name='bilevelmri',
    version='0.1.0',
    packages=find_packages(),
    author='Ferdia Sherry',
    author_email='fs436@cam.ac.uk',
    url='https://github.com/fsherry/bilevelmri',
    license='BSD 2-Clause License',
    project_urls={'arXiv preprint': 'https://arxiv.org/abs/1906.08754'},
    install_requires=['numpy', 'scipy', 'torch'],
    extras_require={'wavelets': 'pytorch_wavelets'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'
    ])

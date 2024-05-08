from setuptools import setup, find_packages

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='BField_LSQ',
    version='0.1',
    description='Mu2e BField model least-squares fit.',
    url='https://github.com/Mu2e/FMS_BFieldModel/BField_LSQ',
    author='Cole Kampa',
    author_email='ckampa13@gmail.com',
    license_files = ('LICENSE',),
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
      'numpy',
      'scipy',
      'pandas',
      'lmfit'],
    include_package_data=True,
    zip_safe=False)

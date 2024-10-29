import sys
import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
# relic from helicalc
#sys.path.append(os.path.dirname(__file__)+'/scripts/')
#from cfgs import run_cfgs

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        #run_cfgs()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        #run_cfgs()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BFieldPINN",
    version="0.0.0",
    author="Cole Kampa",
    author_email="ckampa13@gmail.com",
    description="TensorFlow-based physics-informed neural network (PINN) tools for modeling magnetic fields. A hard constraint on curl and soft constraint on divergence are implemented.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ckampa13/BField_MaPINNg/BFieldPINN",
    packages=setuptools.find_packages(),
    # FIXME! Make this package list complete
    install_requires=[
        "numpy", "scipy", "pandas", "matplotlib"
    ],
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    #scripts=['scripts/SolCalc_GUI/SolCalcGUI.py']
)



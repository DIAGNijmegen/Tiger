import setuptools

import tiger

# Package configuration
setuptools.setup(
    name="tiger",
    description="Medical image analysis library",
    author="Nikolas Lessmann",
    packages=["tiger"],
    version=tiger.__version__,
    install_requires=[
        "numpy",
        "simpleitk>=2.0.0",
        "itk",
        "vtk",
        "tifffile",
        "pillow>=8.1.1",
        "scipy>=1.6.0",
        "scikit-image>=0.17.0",
        "matplotlib",
        "panimg>=0.4.0",
        "gcapi>=0.6.3",
        "python-dateutil",
        "nibabel",
    ],
    extras_require={
        "docs": ["sphinx>=3.0.0", "sphinx_rtd_theme", "sphinx-prompt"],
        "tests": ["pytest", "pytest-cov"],
        "visdom": [
            "visdom @ https://github.com/facebookresearch/visdom/archive/caf411986b59734ef75d24c76da92ad8aedda086.zip",
        ],
        "torch": ["torch>=1.0"],
    },
    python_requires=">=3.8",
    include_package_data=True,
)

from setuptools import setup, find_packages

setup(
    package_dir={"": "src"},
    packages, packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.40.0",
        "semantic-router[hybrid]>=0.0.50",
        "sentence-transformers>=3.0.0"
    ],
    extras_require={
        "dev": ["pytest>=8.0", "black>=24.0"],
        "test": ["pytest>=8.0", "pytest-mock>=3.0"]
    },
    entry_points={
        "console_scripts": ["llmsays=llmsays:cli"]
    }
)
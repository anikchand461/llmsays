#!/usr/bin/env python
"""
setup.py: Legacy setup for llmsays (uses pyproject.toml for modern builds).
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        use_scm_version=True,  # Dynamic version from git tags
        setup_requires=["setuptools_scm"],
        python_requires=">=3.10",
        install_requires=[
            "semantic-router[hybrid]",
            "sentence-transformers",
            "huggingface-hub",  # For model downloads
            "pytest",
            "openai",
        ],
        extras_require={
            "dev": ["pytest>=8.0", "black>=24.0", "flake8>=7.0", "build>=1.0", "twine>=5.0"],
            "test": ["pytest>=8.0", "pytest-mock>=3.0"],
        },
        entry_points={
            "console_scripts": [
                "llmsays=llmsays:cli",
            ],
        },
    )
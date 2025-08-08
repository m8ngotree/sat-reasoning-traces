from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="sat-rl-environment",
    version="1.0.0",
    author="m8ngotree",
    author_email="m8ngotree@gmail.com",
    description="A comprehensive system for building SAT reasoning datasets for large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m8ngotree/sat-rl-environment",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sat-generator=main:main",
        ],
    },
    keywords="sat solver reasoning dataset machine learning ai",
    project_urls={
        "Bug Reports": "https://github.com/m8ngotree/sat-rl-environment/issues",
        "Source": "https://github.com/m8ngotree/sat-rl-environment",
        "Documentation": "https://github.com/m8ngotree/sat-rl-environment#readme",
    },
)
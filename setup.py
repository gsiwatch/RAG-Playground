from setuptools import setup, find_packages

setup(
    name="rag_strategies",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "fastapi>=0.109.2",          
        "uvicorn>=0.27.1",
        "python-dotenv>=1.0.1",    
        "beautifulsoup4>=4.12.3",  
        "pydantic-settings>=2.1.0",
        "pydantic>=2.6.1",           
        "langchain>=0.1.5",          
        "langchain-openai>=0.0.5",   
        "langchain-core>=0.1.14",
        "motor>=3.3.2",
        "pymongo>=4.6.1",            
        "openai>=1.12.0",            
        "aiohttp>=3.9.3",            
        "numpy>=1.26.3",
        "urllib3>=2.4.0",
        "setuptools>=69.0.3",
        "tqdm>=4.66.1"
    ],
     extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    python_requires=">=3.12",
    description="RAG Strategies for document processing and QA",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)

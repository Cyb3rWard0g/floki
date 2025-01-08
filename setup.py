from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="floki-ai",
    version="0.8.1",
    author="Roberto Rodriguez",
    description="Agentic Workflows Made Simple",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cyb3rWard0g/floki",
    project_urls={
        "Documentation": "https://github.com/Cyb3rWard0g/floki",
        "Code": "https://github.com/Cyb3rWard0g/floki",
        "Issue tracker": "https://github.com/Cyb3rWard0g/floki/issues",
    },
    keywords="LLM Cybersecurity AI Agents",
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    install_requires=[
        "pydantic==2.10.2",
        "openai==1.56.1",
        "openapi-pydantic==0.5.0",
        "openapi-schema-pydantic==1.2.4",
        "regex>=2023.12.25",
        "neo4j==5.26.0",
        "Jinja2==3.1.5",
        "azure-identity==1.19.0",
        "dapr==1.14.0",
        "dapr-ext-fastapi==1.14.0",
        "dapr-ext-workflow==0.5.0",
        "colorama==0.4.6",
        "cloudevents==1.11.0"
    ],
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: OS Independent',
        'Topic :: Security',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
)
from setuptools import find_packages, setup


BASE_REQUIREMENTS = []


setup(
    name="nlg_eval_via_simi_measures",
    version="0.1.0dev",
    url="https://github.com/PierreColombo/nlg_eval_via_simi_measures",
    author="Pierre Colombo, Guillaume Staerman",
    author_email="pierre.colombo@centralesupelec.fr, guillaume.staerman@telecom-paris.fr",
    short_description="Automatic evaliation NLG metrics.",
    packages=find_packages(),
    keywords=["natural language generation", "evaluation", "metrics"],
    python_requires=">=3.6",
    install_requires=BASE_REQUIREMENTS,
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={"console_scripts": ["donalg-cli = donalg.cli.cli:main"]},
)

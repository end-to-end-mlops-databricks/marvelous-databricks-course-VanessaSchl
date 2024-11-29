from setuptools import setup, find_packages

import src

setup(
  name = "hotel_reservations",
  version = "3.0.0",
  author = "Vanessa Schlotzig",
  url = "https://www.swk.de/",
  author_email = "vanessa.schlotzig@swk.de",
  description = "Hotel reservation project",
  packages=find_packages(where='./src'),
  package_dir={'': 'src'},
  entry_points={
    "packages": [
      "main=hotel_reservations.main:main"
    ]
  },
  install_requires=[
    "setuptools"
  ]
)
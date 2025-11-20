"""Setup file for BetaMove project."""

from setuptools import find_packages, setup

setup(
    name="betamove",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "mediapipe==0.10.9",
        "ultralytics==8.0.196",
        "xgboost",
        "scikit-learn",
        "fastapi",
        "uvicorn",
    ],
    python_requires=">=3.10",
)

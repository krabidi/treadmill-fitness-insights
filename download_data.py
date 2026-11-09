#!/usr/bin/env python3
"""Download the treadmill exercise-test dataset from PhysioNet into data/.

Mongin D, García Romero J, Alvero Cruz JR. Treadmill Maximal Exercise Tests from the Exercise
Physiology and Human Performance Lab of the University of Malaga (version 1.0.1). PhysioNet, 2021.
https://physionet.org/content/treadmill-exercise-cardioresp/1.0.1/

The data are licensed CC BY-NC-SA 4.0 and are not redistributed in this repository.
"""
import os
import urllib.request

BASE = "https://physionet.org/files/treadmill-exercise-cardioresp/1.0.1/"
FILES = ["subject-info.csv", "test_measure.csv"]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            print(f"{path} already exists, skipping.")
            continue
        print(f"Downloading {BASE + name} (test_measure.csv is about 23 MB)")
        urllib.request.urlretrieve(BASE + name, path)
    print("Done.")


if __name__ == "__main__":
    main()

#!/bin/bash

if [ ! -d "Data" ]; then

    mkdir Data/

    wget https://zenodo.org/record/841984/files/wili-2018.zip?download=1

    unzip ./wili-2018.zip?download=1 -d Data/

    rm ./wili-2018.zip?download=1

fi
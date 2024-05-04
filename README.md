# FIT3162_MCS6_NILM
 This is a collaborative repository for FIT3162 Final Year Project about Non-Intrusive Load Monitoring

## Software Introduction

### Introduction

Our NILM system, which can be downloaded or cloned in our GitHub repository, provides a convenient way to monitor and analyze the energy consumption of individual appliances in your building or household without having to install separate sensors on each appliance. The system uses advanced machine learning techniques known as Non-Intrusive Load Monitoring (NILM) to disaggregate the total energy consumption data into the consumption patterns of individual appliances. In the real world, such a system would be implemented into a smart meter that can be used to detect and predict which appliances are responsible for which power consumption values from the readings. As such, our web application merely serves as a prototype that anyone can use to test out how a smart meter would work.

### Getting Started
After setting up the environment and running the StreamLit application locally as seen in the steps in the Software Installation section, you will need to upload an H5 file containing the main energy consumption data, which includes the Active (W) and Apparent (VA) power, for your building over a given time period, specifically in UNIX time format. In a real-world use case, the smart meter would be programmed to read the power consumption values and proceed from there automatically. However, in our prototype, you will be provided with a test dataset that you can download and input into our model to test out our system.
Once you have downloaded and uploaded the H5 file, you can then run it through our state-of-the-art, time series Seq2Point model, which has been trained to recognize the consumption patterns of a wide variety of common household appliances (specifically in Malaysia). The model will then generate predicted power consumption values for each appliance it has been trained on, which includes the following 8 appliances: fridge, air conditioner, washing machine, tumble dryer, kettle, vacuum cleaner, electric water heating appliance, and oven.

### Visualizing the Results
After the model has processed your data, you will be prompted to select which specific appliances you want via a dropdown menu to visualize from the previously mentioned ones. The application will then generate a line graph, displaying the predicted power consumption over time for your selected appliances. Moreover, the on-off states of each appliance will also be clearly depicted through a binary line graph, where 1 would mean that the appliance is turned on and 0 otherwise. Aside from selecting appliances, the application also includes a feature that allows you to adjust the time range of the output graphs. With this power tool, you can gain valuable insights into your energy usage patterns without the hassle and expense of instrumenting each appliance. Identify energy hogs, optimize usage, and most importantly, save money on your utility bills!


## How to Use the Application

After setting up your environment and running the StreamLit application locally on your device, you will now be able to use the application via the following steps:

1. Upload the test dataset by either dragging it into the “Drag and drop file here” area or by clicking on the “Browse files” button and selecting the dataset. The test dataset can be found as “mimos_1sec.h5” under the data folder.

2. Now please wait for the model to finish running. Once the model finishes running, a sidebar will pop up from the left side of your screen with a dropdown menu containing the various detected appliances and two sliders that allow you to adjust the time range of the output graphs.

3. Now, you can click on each appliance button, and the application will display two graphs for each appliance:
    - Generated power consumption values for each appliance.
    - The ON/OFF state of each appliance.

## Software Installation

Environment Setup Details
We are testing the models from the [NILMTK-Contrib](https://github.com/nilmtk/nilmtk-contrib) using the public [NILMTK API](https://github.com/nilmtk/nilmtk). You can install the conda environment directly in Anaconda with the following command:
```
conda install -c conda-forge -c nilmtk nilmtk-contrib
```
OR create a dedicated environment (recommended) in Anaconda with the following command:
```
conda create -n nilm -c conda-forge -c nilmtk nilmtk-contrib
```
Note: You may refer to the “nilm.ipynb” and “using_imported_models.ipynb” files for using the algorithms from the NILMTK-Contrib, using the NILMTK API.

After creating the environment, activate the conda environment.
```
conda activate nilm
```

Then, run the following commands in your conda terminal:
```
pip install git+https://github.com/nilmtk/nilm_metadata
pip install --upgrade pandas
pip install tensorflow== 2.11.0 
pip install streamlit==1.22.0
```
Note: You may ignore any warning messages that pop up during this setup phase.

MCS6 NILM Repository
Our main GitHub NILM repository contains all the following folders and files:
CSV and H5 files containing various versions of our dataset
Python files and Jupyter notebooks that we used for data-pre-processing
Folder containing our model performances
Jupyter notebooks for the functionality of our NILM system
Includes the functions for running the time series models (Seq2Seq and Seq2Point) from the NILMTK-Contrib.
The Python file to our streamlit application

### Repository Installation Details
You can simply clone or fork our main GitHub NILM repository and start experimenting with it.

If you want to run your own streamlit web application locally, then run the following command in your terminal:
```
streamlit run [your_script].py
```

### Dependencies
Make sure you have installed and are using the correct versions of the following packages:
- numpy == 1.21.6 
- pandas == 1.3.5 
- tensorflow == 2.11.0 
- protobuf == 3.19.6 
- streamlit == 1.22.0 
- nilmtk == 0.4.0 
- nilmtk-contrib == 0.1.1 
- nilm-metadata == 0.2.5 
- python == 3.7.*



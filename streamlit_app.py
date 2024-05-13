import streamlit as st
import os
from os.path import join
import pickle
import nilmtk as ntk
from api import API
from tempfile import NamedTemporaryFile
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Import the model
def import_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

# Pre-generate graphs for all appliances
def pre_generate_graphs():
    for appliance in st.session_state.appliances:
        chart_data = st.session_state.model.pred_overall['Seq2SPoint'][appliance]
        X = chart_data.values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        threshold = (centroids[0][0] + centroids[1][0]) / 2
        st.session_state.model.pred_overall['Seq2SPoint'][appliance + ' ON/OFF states'] = 0
        window_size = 99
        for k in range(0, len(st.session_state.model.pred_overall['Seq2SPoint']), window_size - 1):
            if st.session_state.model.pred_overall['Seq2SPoint'][appliance].iloc[k] > threshold:
                st.session_state.model.pred_overall['Seq2SPoint'][appliance + ' ON/OFF states'].iloc[k:k + window_size] = 1

# Display selected appliance information
def display_appliance_info(appliance, start_time, end_time):
    st.subheader(appliance.capitalize() + " Usage with ON/OFF states")
    chart_data = st.session_state.model.pred_overall['Seq2SPoint'][appliance].loc[start_time:end_time]
    chart_data.index = chart_data.index.strftime('%H:%M:%S')
    st.line_chart(chart_data)
    
    on_off_chart = st.session_state.model.pred_overall['Seq2SPoint'][appliance + ' ON/OFF states'].loc[start_time:end_time]
    on_off_chart.index = on_off_chart.index.strftime('%H:%M:%S')
    st.line_chart(on_off_chart)

# Validate timestamp format
def validate_timestamp(timestamp_str):
    if not timestamp_str:
        return False
    try:
        pd.to_datetime(timestamp_str)
        return True
    except ValueError:
        return False
    
def validate_h5_file(filename):
    try:
        ds = ntk.DataSet(filename)
        errors = []

        for i in ds.buildings:
            mains = ds.buildings[i].elec.mains().available_ac_types('power')
            submeters = ds.buildings[i].elec.submeters().available_ac_types('power')
            if mains != ['active']:
                errors.append(f"The power type for main meter in building {i} should be Active(W) only.")
            if submeters != ['active']:
                errors.append(f"The power type for all appliances (sub-meters) in building {i} should be Active(W) only.")
        
        if errors:
            error_message = "\n".join(errors)
            raise Exception

        return True
    
    except Exception as e:
        raise Exception("Invalid HDF5 file format. Please upload a valid file.")

def main():
    # The UI Title
    st.title('Non-Intrusive Load Monitoring')

    st.markdown("""
    <style>
        .stButton>button {
            width: 150px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Upload and store the input file
    uploaded_file = st.file_uploader("Insert your HDF5 file here (e.g. YOUR_FILE.h5)", accept_multiple_files=False, type = ['h5'])
    
    model = None
    
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None
    
    if 'appliances' not in st.session_state:
        st.session_state.appliances = None
    
    if uploaded_file:
        
        with NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            if validate_h5_file(temp_file_path):
                alert = st.success("HDF5 file validation passed")

                test_params = {
                    # Insert the path to the file from the uploaded file
                    'path': temp_file_path,
                    'buildings':{
                        5:{
                            'start_time': '2022-11-08',
                            'end_time': '2022-11-09'
                        }
                    }
                }
                
                if st.session_state.loaded == False:
                    with st.spinner('Running the model...'):
                        # The model path
                        trained_model_path = join(os.getcwd(), "trained_models/1sec_99SL.pickle")
                        
                        # Import the model
                        model = import_model(trained_model_path)

                        # Test the model using only Seq2Point
                        model.test_jointly({'test': test_params})
                        
                        st.session_state.model = model
                        st.session_state.loaded = True
                
                if st.session_state.dataframe is None and st.session_state.appliances is None:
                    st.session_state.dataframe = st.session_state.model.pred_overall['Seq2SPoint'].copy()
                    st.session_state.appliances = st.session_state.dataframe.columns.tolist()
                    # Pre-generate graphs for all appliances
                    pre_generate_graphs()
            
            alert.empty()

            if uploaded_file and st.session_state.loaded:
                # Sidebar with buttons and slider
                st.sidebar.markdown("### Appliances")
                selected_appliance = st.sidebar.selectbox("Select Appliance", st.session_state.appliances)

                st.sidebar.markdown("### Adjust Time Range")
                start_header = "Start Time (Default: " + pd.Timestamp.fromtimestamp(st.session_state.model.pred_overall['Seq2SPoint'].index.min().timestamp()).strftime('%Y-%m-%d %H:%M:%S') + ")"
                start_time_input = st.sidebar.text_input(start_header, value=pd.Timestamp.fromtimestamp(st.session_state.model.pred_overall['Seq2SPoint'].index.min().timestamp()).strftime('%Y-%m-%d %H:%M:%S'))
                
                end_header = "End Time (Default: " + pd.Timestamp.fromtimestamp(st.session_state.model.pred_overall['Seq2SPoint'].index.max().timestamp()).strftime('%Y-%m-%d %H:%M:%S') + ")"
                end_time_input = st.sidebar.text_input(end_header, value=pd.Timestamp.fromtimestamp(st.session_state.model.pred_overall['Seq2SPoint'].index.max().timestamp()).strftime('%Y-%m-%d %H:%M:%S'))

                # Validate start time input
                if not validate_timestamp(start_time_input):
                    st.error("Error: Please enter a valid start time format (YYYY-MM-DD HH: MM: SS).")
                    st.error("Hours should be in 24-hour format.")
                    st.error("Minutes should be within 0-59.")
                    st.error("Seconds should be within 0-59.")
                    return

                start_time_readable = pd.Timestamp(start_time_input)
                end_time_readable = pd.Timestamp(end_time_input)
                    

                # Validation check for start and end times
                if start_time_readable > end_time_readable:
                    st.error("Error: Please select a valid time range. Start time cannot be greater than end time.")
                    return
                
                if start_time_readable < pd.Timestamp.fromtimestamp(st.session_state.model.pred_overall['Seq2SPoint'].index.min().timestamp()) or end_time_readable > pd.Timestamp.fromtimestamp(st.session_state.model.pred_overall['Seq2SPoint'].index.max().timestamp()):
                    st.error("Error: Please enter a valid time range that is within the default start and end time.")
                    return

                display_appliance_info(selected_appliance, start_time_readable, end_time_readable)
        
        except Exception as e:
            st.error("Error: " + str(e))
            return

if __name__ == "__main__":
    main()

import streamlit as st
import os
from os.path import join
import pickle
from api import API
from tempfile import NamedTemporaryFile
from sklearn.cluster import KMeans
import pandas as pd

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
    st.line_chart(chart_data)
    on_off_chart = st.session_state.model.pred_overall['Seq2SPoint'][appliance + ' ON/OFF states'].loc[start_time:end_time]
    st.line_chart(on_off_chart)

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
    
    if uploaded_file and st.session_state.loaded:
        # Sidebar with buttons and slider
        st.sidebar.markdown("### Appliances")
        selected_appliance = st.sidebar.selectbox("Select Appliance", st.session_state.appliances)

        st.sidebar.markdown("### Adjust Time Range (UNIX)")
        start_time_unix = st.sidebar.slider("Start Time", min_value=st.session_state.model.pred_overall['Seq2SPoint'].index.min().timestamp(), max_value=st.session_state.model.pred_overall['Seq2SPoint'].index.max().timestamp(), value=st.session_state.model.pred_overall['Seq2SPoint'].index.min().timestamp())
        end_time_unix = st.sidebar.slider("End Time", min_value=st.session_state.model.pred_overall['Seq2SPoint'].index.min().timestamp(), max_value=st.session_state.model.pred_overall['Seq2SPoint'].index.max().timestamp(), value=st.session_state.model.pred_overall['Seq2SPoint'].index.max().timestamp())
        start_time = pd.Timestamp.fromtimestamp(start_time_unix)
        end_time = pd.Timestamp.fromtimestamp(end_time_unix)
        
        # Validation check for start and end times
        if start_time > end_time:
            st.sidebar.error("Error: Please select a valid time range. Start time cannot be greater than end time.")
        else:
            display_appliance_info(selected_appliance, start_time, end_time)

if __name__ == "__main__":
    main()

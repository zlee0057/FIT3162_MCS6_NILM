import streamlit as st
import os
from os.path import join
import pickle
from api import API
from tempfile import NamedTemporaryFile
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import altair as alt

# Import the model
def import_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


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
    
    if uploaded_file:
        
        with NamedTemporaryFile(delete = False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
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
            with st.spinner('Loading...'):
                # The model path
                trained_model_path = join(os.getcwd(), "trained_models/1sec_99SL.pickle")
                
                # Import the model
                model = import_model(trained_model_path)

                # Test the model using only Seq2Point
                model.test_jointly({'test': test_params})
                
                st.session_state.model = model
                st.session_state.loaded = True
                
        dataframe = st.session_state.model.pred_overall['Seq2SPoint']
        appliances = dataframe.columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        columns = [col1, col2, col3]
        
        num_appliance = len(appliances)
        
        k = 0
        for i in range(num_appliance):
            button_key = f"button_{i}"
            if columns[k % 3].button(appliances[i].capitalize(), key = button_key):
                
                # st.subheader(appliances[i].capitalize() + " Usage")
                
                # Extract the values
                X = st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]].values.reshape(-1, 1)

                # Apply k-means clustering with k=2
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(X)
                
                # Get the centroids
                centroids = kmeans.cluster_centers_

                # Calculate the threshold value
                threshold = (centroids[0][0] + centroids[1][0]) / 2
                
                
                base = alt.Chart(st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]]).mark_line().encode(
                    x=st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]].index,
                    y=alt.Y(appliances[i], title='Power Usage')
                )

                # Conditional coloring
                line = base.encode(
                    color=alt.condition(
                        alt.datum.y > threshold,
                        alt.value('red'),  # Color if value is greater than threshold
                        alt.value('blue')  # Color if value is less than or equal to threshold
                    )
                )

                # Plot the chart
                st.altair_chart(line, use_container_width=True)
                
                
                # chart_data = st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]]
                # st.line_chart(chart_data)
            
                
                # # Plot the line graph
                # plt.figure(figsize = (15, 9))
                
                # plt.plot(st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]].index, st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]], color='blue')
                
                # # Iterate through rows in new_data to plot background colors
                # for j in range(1, len(st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]])):
                #     if st.session_state.model.pred_overall['Seq2SPoint'][appliances[i]][j] > threshold:
                #         plt.axvspan(st.session_state.model.pred_overall['Seq2SPoint'].index[j-1], st.session_state.model.pred_overall['Seq2SPoint'].index[j], color='orange', alpha=0.05)
                
                # # Set labels and title
                # plt.xlabel('Timestamp')
                # plt.ylabel('Power Usage')
                # plt.title(appliances[i].capitalize() + " Usage with ON-OFF states")

                # # Show plot
                # st.pyplot(plt)
                
            k += 1
    
    else:
        st.session_state.loaded = False



if __name__ == "__main__":
    main()
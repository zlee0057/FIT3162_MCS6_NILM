import pytest
import pandas as pd
import os
from os.path import join
import tempfile
from seleniumbase import BaseCase
from selenium.webdriver.common.by import By
from streamlit_app import validate_timestamp, validate_h5_file

@pytest.mark.parametrize("building_number", range(1, 8))
def test_merge_main_and_appliance_files(building_number):
    def merge_main_and_appliance_files(main_csv_path, appliance_csv_path):
        main_df = pd.read_csv(main_csv_path)
        appliances_df = pd.read_csv(appliance_csv_path)
        merged_df = pd.merge(main_df, appliances_df, on='Timestamp', how='outer')
        merged_df.fillna(0, inplace=True)
        return merged_df
    
    # Paths to the test CSV files
    main_csv_path = f'data_preprocessing/buildings_by_date/Building {building_number}/building_{building_number}_main_output.csv'
    appliance_csv_path = f'data_preprocessing/buildings_by_date/Building {building_number}/appliances_by_column.csv'
    
    # Read the CSV files
    main_df = pd.read_csv(main_csv_path)
    appliance_df = pd.read_csv(appliance_csv_path)
    
    # Merge main and appliances
    merged_df = merge_main_and_appliance_files(main_csv_path, appliance_csv_path)
    
    # Check missing columns
    expected_columns = ['Timestamp', 'Active (W)', 'Apparent (VA)', 'kettle', 'vacuum', 'water_heater', 'oven', 'fridge', 'washing_machine', 'dryer', 'aircond']
    missing_columns = [col for col in expected_columns if col not in merged_df.columns]
    assert not missing_columns, f"Missing columns: {missing_columns}"
    
    # Check first row values
    first_row_main = main_df.iloc[0]
    first_row_appliance = appliance_df.iloc[0]
    first_row_merged = merged_df.iloc[0]
    
    for column in main_df.columns:
        assert first_row_merged[column] == first_row_main[column], f"Mismatch in column {column} for first row in Building {building_number}."

    for column in appliance_df.columns:
        if column != 'Timestamp':
            assert first_row_merged[column] == first_row_appliance[column], f"Mismatch in column {column} for first row in Building {building_number}."

    # Check last row appliance values
    last_row_appliance = appliance_df.iloc[-1]
    last_row_merged = merged_df.iloc[-1]

    for column in appliance_df.columns:
        if column != 'Timestamp':
            assert last_row_merged[column] == last_row_appliance[column], f"Mismatch in column {column} for last row in Building {building_number}."

@pytest.mark.parametrize("sampling_period", ['3s', '6s', '30s'])
def test_resample_csv(sampling_period):
    def resample_csv(input_path, output_path, sampling_period='6s'):
        """Resample the dataset to a new frequency.

        Parameters:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the resampled CSV file.
        sampling_period (str): New sampling frequency (default is '6s' for 6 seconds).
        """

        # Load the dataset
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)

        # Resample the dataset
        resampled_df = df.resample(sampling_period).mean()

        # Drop rows where all columns (except for the index) are NaN
        resampled_df = resampled_df.dropna(how='all')

        # Save the resampled dataset
        resampled_df.to_csv(output_path)

    # Path to the real input CSV file
    input_file_path = join(os.getcwd(), "data_preprocessing/csv_files/1_sec/Building_1.csv")
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as output_file:
        # Call the resample_csv function
        resample_csv(input_file_path, output_file.name, sampling_period)
        
        # Read the output file
        resampled_df = pd.read_csv(output_file.name, parse_dates=['Timestamp'])
    
    # Read the input file
    df = pd.read_csv(input_file_path)
    
    # Calculate the number of rows to average based on the sampling period
    period_seconds = int(sampling_period.rstrip('s'))
    
    # Expected average values for the first resampling period
    expected_row = {
        'dryer': df['dryer'][:period_seconds].mean(),
        'kettle': df['kettle'][:period_seconds].mean(),
        'vacuum': df['vacuum'][:period_seconds].mean(),
        'water_heater': df['water_heater'][:period_seconds].mean(),
        'oven': df['oven'][:period_seconds].mean(),
        'fridge': df['fridge'][:period_seconds].mean(),
        'washing_machine': df['washing_machine'][:period_seconds].mean(),
        'aircond': df['aircond'][:period_seconds].mean()
    }

    # Assertion to check if the first row is the expected average
    for column, expected_value in expected_row.items():
        # Check if the column is NaN
        assert pd.isna(resampled_df.iloc[0][column]) == pd.isna(expected_value), f"Column {column} NaN mismatch"
        # Check if the column average is approximately equal to the expected value
        if not pd.isna(expected_value):
            assert resampled_df.iloc[0][column] == pytest.approx(expected_value), f"Column {column} average mismatch"
    
@pytest.mark.parametrize("input, expected", [
    # Valid timestamp
    ("2024-05-10 14:23:00", True),
    # Valid timestamp for a leap year
    ("2024-02-29 00:00:00", True),
    # Invalid timestamp, regular year
    ("2023-02-29 00:00:00", False),
    # Invalid time, hour should not exceed 23
    ("2024-12-01 24:01:00", False),
    # Invalid day, day should be between 01 and 31
    ("2024-12-00 12:00:00", False),
    # Invalid day, December has only 31 days
    ("2024-12-32 12:00:00", False),
    # Invalid date, April has only 30 days
    ("2024-04-31 00:00:00", False),
    # Invalid date, month should be between 01 and 12
    ("2024-00-12 12:00:00", False),
    # Invalid date, month should be between 01 and 12
    ("2024-13-12 12:00:00", False),
    # Invalid timestamp, not a timestamp
    ("not-a-timestamp", False),
    # Invalid timestamp, empty string
    ("", False),
    # Valid timestamp, last datetime of the year
    ("2024-12-12 23:59:59", True),
    # Valid timestamp, first datetime of the year
    ("2024-01-01 00:00:00", True)
])
def test_validate_timestamp(input, expected):
    assert validate_timestamp(input) == expected

def test_validate_h5_file():
    # Test the function with a valid HDF5 file
    assert validate_h5_file('./data/mimos_1_sec.h5')

# Integration test for the Streamlit app
class StreamlitAppTests(BaseCase):

    def setUp(self):
        super().setUp()

        # Open the Streamlit app URL
        self.open("http://localhost:8501")

        # Upload the HDF5 file
        file_path = './data/mimos_1_sec.h5'
        self.choose_file("input[type='file']", file_path)

    def tearDown(self):
        super().tearDown()

    def test_upload(self):
        # Wait for the HDF5 file validation message
        self.assert_text("HDF5 file validation passed", 'p:contains("HDF5 file validation passed")', timeout=10)

    def test_generate_graphs(self):
        # Wait for the model to finish running
        self.wait_for_element('p:contains("Running the model...")')
        self.wait_for_element_absent('p:contains("Running the model...")', timeout=50)

        # Wait for the graphs to be generated
        self.wait_for_element('p:contains("Loading...")')
        self.wait_for_element_absent('p:contains("Loading...")', timeout=10)

        # Check if the graphs are generated
        charts = self.find_elements(""".//*[@data-testid="stArrowVegaLiteChart"]//*[contains(concat(" ",normalize-space(@class)," ")," chart-wrapper ")]//*[contains(concat(" ",normalize-space(@class)," ")," marks ")]""", by=By.XPATH)
        
        # Check if the number of generated graphs is correct
        assert len(charts) == 2
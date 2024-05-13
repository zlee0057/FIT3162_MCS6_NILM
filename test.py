import pytest
from seleniumbase import BaseCase
from selenium.webdriver.common.by import By
from streamlit_app import validate_timestamp, validate_h5_file


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
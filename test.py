import pytest
from streamlit_app import validate_timestamp


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

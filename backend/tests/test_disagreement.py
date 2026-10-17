import pytest

from app.domain.disagreement import FlagCode, disagreement


@pytest.mark.parametrize(
    ("priority_open", "is_high", "expected"),
    [
        (0, False, FlagCode.NONE),
        (0, True, FlagCode.ML_HIGH_CRITERIA_MET),
        (1, False, FlagCode.CRITERIA_OPEN_ML_LOW),
        (1, True, FlagCode.BOTH_REVIEW),
    ],
)
def test_disagreement_matrix(priority_open, is_high, expected):
    code, _ = disagreement(priority_open, is_high)

    assert code == expected

import pytest

from md_reheader.eval.metrics import (
    compute_all_metrics,
    exact_match,
    hierarchy_preservation,
    level_count_match,
    mean_absolute_error,
    per_heading_accuracy,
)


class TestExactMatch:
    def test_perfect(self):
        assert exact_match([1, 2, 2, 3], [1, 2, 2, 3]) == 1.0

    def test_one_off(self):
        assert exact_match([1, 2, 2, 2], [1, 2, 2, 3]) == 0.0

    def test_length_mismatch(self):
        assert exact_match([1, 2], [1, 2, 3]) == 0.0

    def test_empty(self):
        assert exact_match([], []) == 1.0


class TestPerHeadingAccuracy:
    def test_perfect(self):
        assert per_heading_accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_none_correct(self):
        assert per_heading_accuracy([3, 3, 3], [1, 2, 1]) == 0.0

    def test_partial(self):
        assert per_heading_accuracy([1, 2, 3], [1, 2, 1]) == pytest.approx(2 / 3)

    def test_length_mismatch(self):
        assert per_heading_accuracy([1, 2], [1, 2, 3]) == 0.0


class TestHierarchyPreservation:
    def test_correct_relative_structure(self):
        pred = [2, 3, 3, 4, 3]
        truth = [1, 2, 2, 3, 2]
        assert hierarchy_preservation(pred, truth) == 1.0

    def test_inverted_relationship(self):
        pred = [1, 2, 1]
        truth = [1, 2, 3]
        assert hierarchy_preservation(pred, truth) < 1.0

    def test_single_heading(self):
        assert hierarchy_preservation([1], [1]) == 0.0

    def test_length_mismatch(self):
        assert hierarchy_preservation([1, 2], [1, 2, 3]) == 0.0


class TestMeanAbsoluteError:
    def test_perfect(self):
        assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_off_by_one(self):
        assert mean_absolute_error([2, 3, 4], [1, 2, 3]) == 1.0

    def test_length_mismatch(self):
        assert mean_absolute_error([1, 2], [1, 2, 3]) == float("inf")


class TestLevelCountMatch:
    def test_match(self):
        assert level_count_match([1, 2, 3], [1, 2, 3]) is True

    def test_mismatch(self):
        assert level_count_match([1, 2], [1, 2, 3]) is False


class TestComputeAllMetrics:
    def test_perfect_prediction(self):
        result = compute_all_metrics([1, 2, 2, 3], [1, 2, 2, 3])
        assert result.exact_match == 1.0
        assert result.per_heading_accuracy == 1.0
        assert result.mean_absolute_error == 0.0
        assert result.level_count_match is True

import pytest

from examples.interact.merge_qwen35_wandb import scalars, points, compare_histories


def test_preserves_zero_and_removes_internal_metadata_and_unrelated_fields():
    assert scalars({'_step': 100, '_runtime': 1, 'train/step': 0, 'train/loss': 0.,
                    'eval/success': None, 'WANDB_API_KEY': 'secret'}) == {
                        'train/step': 0, 'train/loss': 0.}
    with pytest.raises(ValueError, match='non-finite'):
        scalars({'train/loss': float('nan')})


def test_points_preserve_original_coordinates_and_reject_conflicts():
    rows = [{'train/step': 0, 'train/loss': .1}, {'train/step': 2, 'train/loss': .2}]
    assert points(rows, 'train/loss', 'train/step') == {0: .1, 2: .2}
    with pytest.raises(ValueError, match='conflicting'):
        points(rows + [{'train/step': 2, 'train/loss': .3}], 'train/loss', 'train/step')
    with pytest.raises(ValueError, match='has no'):
        points([{'train/loss': .1}], 'train/loss', 'train/step')


def test_readback_detects_missing_changed_and_duplicated_points():
    rows = [{'eval/step': 0, 'eval/success': .5}, {'eval/step': 9, 'eval/success': .6}]
    assert compare_histories(rows, rows) == 2
    for wrong in (rows[:1], rows + rows[-1:], [rows[0], {'eval/step': 9, 'eval/success': .7}]):
        with pytest.raises(ValueError, match='read-back mismatch'):
            compare_histories(rows, wrong)

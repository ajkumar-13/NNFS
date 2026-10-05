"""The data path: the archive check, the parser, the split, and the no-leakage rule of nn-029."""

import numpy as np
import pytest

from california_housing_regression import data
from california_housing_regression.data import (
    ARCHIVE,
    COLUMNS,
    FEATURE_NAMES,
    MEMBER,
    DataError,
    DatasetFile,
    Scaler,
    check_file,
    destandardise_y,
    features_and_target,
    load_california_housing,
    load_table,
    parse_table,
    read_member,
    sha256_of,
    split_indices,
    standardise_apply,
    standardise_fit,
)
from helpers import (
    FIXTURE_ARCHIVE,
    FIXTURE_DIR,
    FIXTURE_ROWS,
    FIXTURE_TEST,
    FIXTURE_TEST_ROWS,
    FIXTURE_TRAIN,
    FIXTURE_TRAIN_ROWS,
    archive_bytes,
    fixture_row,
    fixture_table,
    table_bytes,
    write_archive,
)

# What the project's original data.py produced for the tiny fixture with seed 0, before the code
# was restructured into this package and while it still read the data through scikit-learn.
RECORDED_X_MEAN = [
    *[5.778749942779541, 26.625, 5.78125, 1.1924933195114136],
    *[881.625, 4.424665451049805, 36.57499694824219, -119.20628356933594],
]
RECORDED_X_STD = [
    *[2.3815274238586426, 16.427396774291992, 1.4083318710327148, 0.07096507400274277],
    *[351.5588684082031, 0.6071448922157288, 2.2681214809417725, 2.8351521492004395],
]
RECORDED_Y_MEAN = 3.110938787460327
RECORDED_Y_STD = 1.3646493958068848
RECORDED_X_TRAIN_ROW = [
    *[-0.20942430198192596, -0.22066795825958252, 0.8653855323791504, 0.10578060150146484],
    *[-0.20942439138889313, 0.10577917844057083, -0.2094237357378006, -0.20941361784934998],
]
RECORDED_Y_TRAIN = [-0.19121313095092773, -1.748389482498169, -0.9240019917488098]
RECORDED_X_TEST_ROW = [
    *[-0.8266753554344177, -0.03804619982838631, -0.5547342896461487, -0.5744969844818115],
    *[-0.8266751766204834, -0.5744979977607727, -0.8266736268997192, -0.8266644477844238],
]
RECORDED_Y_TEST = [-0.8324033617973328, 0.6331744194030762, -1.4735935926437378]


def random_table(rows=60, seed=5):
    """A valid table of the dataset's nine columns with unrelated random content."""
    generator = np.random.RandomState(seed)
    households = generator.randint(50, 900, size=rows).astype(float)
    return np.column_stack(
        [
            generator.uniform(-124, -114, rows),
            generator.uniform(32, 42, rows),
            generator.randint(1, 53, size=rows).astype(float),
            households * generator.uniform(3, 8, rows),
            households * generator.uniform(0.9, 1.3, rows),
            households * generator.uniform(2, 4, rows),
            households,
            generator.uniform(0.5, 15, rows),
            generator.randint(15_000, 500_001, size=rows).astype(float),
        ]
    ).round(6)


# --------------------------------------------------------------------------- the dataset record


def test_the_record_is_the_published_archive():
    assert (
        DatasetFile(
            "cal_housing.tgz",
            441_963,
            "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681",
        )
        == ARCHIVE
    )
    assert MEMBER == "CaliforniaHousing/cal_housing.data"
    assert len(COLUMNS) == 9
    assert len(FEATURE_NAMES) == 8


def test_the_committed_fixture_is_the_one_the_script_writes():
    path = FIXTURE_DIR / FIXTURE_ARCHIVE.name
    assert path.stat().st_size == FIXTURE_ARCHIVE.size
    assert sha256_of(path) == FIXTURE_ARCHIVE.sha256


# ------------------------------------------------------------------------------------ check_file


def test_check_file_returns_the_path_of_an_unaltered_archive():
    assert check_file(FIXTURE_DIR, FIXTURE_ARCHIVE) == FIXTURE_DIR / "cal_housing.tgz"


def test_check_file_names_the_download_command_when_the_directory_is_missing(tmp_path):
    with pytest.raises(DataError, match=r"does not exist.*\n.*download_california_housing\.py"):
        check_file(tmp_path / "absent", FIXTURE_ARCHIVE)


def test_check_file_names_the_download_command_when_the_archive_is_missing(tmp_path):
    with pytest.raises(DataError, match=r"is missing cal_housing\.tgz.*\n.*download_california"):
        check_file(tmp_path, FIXTURE_ARCHIVE)


def test_check_file_refuses_an_archive_with_another_checksum(tmp_path):
    (tmp_path / "cal_housing.tgz").write_bytes(b"not the dataset")
    with pytest.raises(DataError, match="is not the expected file: its SHA-256 is"):
        check_file(tmp_path, FIXTURE_ARCHIVE)


def test_the_real_record_refuses_the_fixture():
    with pytest.raises(DataError, match="is not the expected file"):
        load_table(FIXTURE_DIR)


# ----------------------------------------------------------------------------------- read_member


def test_read_member_returns_the_table_bytes():
    raw = read_member(FIXTURE_DIR / "cal_housing.tgz")
    lines = raw.decode("ascii").splitlines()
    assert len(lines) == FIXTURE_ROWS
    assert lines[0] == (
        "-124.300000,32.500000,1.000000,360.000000,90.000000,250.000000,90.000000,1.500000,"
        "60000.000000"
    )


def test_read_member_rejects_a_file_that_is_not_an_archive(tmp_path):
    path = tmp_path / "cal_housing.tgz"
    path.write_bytes(b"plain text, not gzip")
    with pytest.raises(DataError, match="is not a readable gzip tar archive"):
        read_member(path)


def test_read_member_rejects_a_truncated_archive(tmp_path):
    path = tmp_path / "cal_housing.tgz"
    whole = (FIXTURE_DIR / "cal_housing.tgz").read_bytes()
    path.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(DataError, match="is not a readable gzip tar archive"):
        read_member(path)


def test_read_member_rejects_an_archive_without_the_table(tmp_path):
    path = tmp_path / "cal_housing.tgz"
    path.write_bytes(archive_bytes({"CaliforniaHousing/other.data": b"1,2,3\n"}))
    with pytest.raises(DataError, match=r"does not contain CaliforniaHousing/cal_housing\.data"):
        read_member(path)


def test_read_member_rejects_a_directory_under_the_table_name(tmp_path):
    path = tmp_path / "cal_housing.tgz"
    path.write_bytes(archive_bytes({}, directories=(MEMBER,)))
    with pytest.raises(DataError, match="is not a regular file"):
        read_member(path)


def test_read_member_rejects_a_table_larger_than_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_TABLE_BYTES", 100)
    with pytest.raises(DataError, match="more than the 100 this project reads"):
        read_member(FIXTURE_DIR / "cal_housing.tgz")


# ----------------------------------------------------------------------------------- parse_table


def test_parse_table_reads_every_cell_of_the_fixture():
    table = parse_table(read_member(FIXTURE_DIR / "cal_housing.tgz"))
    assert table.shape == (FIXTURE_ROWS, 9)
    assert table.dtype == np.float64
    assert np.array_equal(table, fixture_table())
    assert table[7].tolist() == pytest.approx(fixture_row(7))


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"", "is empty"),
        (b"  \n\n", "is empty"),
        (b"1,2,3,4,5,6,7,8,nine\n", "is not a table of comma-separated numbers"),
        (b"1,2,3,4,5,6,7,8,9\n1,2,3\n", "is not a table of comma-separated numbers"),
        (b"1,2,3,4,5,6,7,8\n", "has 8 columns, expected the 9 columns longitude"),
        (b"1,2,3,4,5,6,7,8,9,10\n", "has 10 columns"),
        (b"1,2,3,4,5,6,7,8,nan\n", "contains values that are not finite"),
        (b"1,2,3,4,5,6,7,inf,9\n", "contains values that are not finite"),
        (b"1,2,3,4,5,6,0,8,9\n", "a block group with no households"),
        (b"1,2,3,4,5,6,-3,8,9\n", "a block group with no households"),
    ],
)
def test_parse_table_rejects_what_is_not_the_table(raw, message):
    with pytest.raises(DataError, match=message):
        parse_table(raw, "the test table")


def test_parse_table_names_its_source_in_the_message():
    with pytest.raises(DataError, match=r"^somewhere is empty$"):
        parse_table(b"", "somewhere")


def test_parse_table_accepts_a_single_row():
    assert parse_table(b"1,2,3,4,5,6,7,8,9\n").shape == (1, 9)


# --------------------------------------------------------------------------- features_and_target


def test_features_and_target_on_a_hand_computed_row():
    # longitude, latitude, age, rooms, bedrooms, population, households, income, value
    table = np.array([[-122.25, 37.5, 41.0, 880.0, 110.0, 330.0, 100.0, 8.25, 452600.0]])
    X, y, value = features_and_target(table)
    # MedInc, HouseAge, rooms / households, bedrooms / households, Population,
    # population / households, Latitude, Longitude
    expected = np.array([[8.25, 41.0, 8.8, 1.1, 330.0, 3.3, 37.5, -122.25]], dtype=np.float32)
    assert np.array_equal(X, expected)
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert y[0] == np.float32(4.526)
    assert value.dtype == np.float64
    assert value[0] == 452600.0


def test_features_are_rounded_to_float32_once_after_float64_arithmetic():
    table = random_table()
    X, y, value = features_and_target(table)
    rooms, bedrooms, population, households, price = table[:, [3, 4, 5, 6, 8]].T
    assert np.array_equal(X[:, 2], (rooms / households).astype(np.float32))
    assert np.array_equal(X[:, 3], (bedrooms / households).astype(np.float32))
    assert np.array_equal(X[:, 5], (population / households).astype(np.float32))
    assert np.array_equal(X[:, [0, 1, 4, 6, 7]], table[:, [7, 2, 5, 1, 0]].astype(np.float32))
    assert np.array_equal(y, (price / 100_000.0).astype(np.float32))
    assert np.array_equal(value, price)


def test_features_and_target_leave_the_table_untouched():
    table = random_table()
    before = table.copy()
    _, _, value = features_and_target(table)
    value[:] = 0
    assert np.array_equal(table, before)


# --------------------------------------------------------------------------------- split_indices


def test_split_is_a_partition_with_a_fifth_in_the_test_fold():
    train, test = split_indices(20_640, 0.2, seed=0)
    # round(20,640 * 0.2) = 4,128 and 20,640 - 4,128 = 16,512.
    assert len(test) == 4_128
    assert len(train) == 16_512
    assert len(np.intersect1d(train, test)) == 0
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(20_640))


def test_split_of_the_fixture_is_the_recorded_one():
    train, test = split_indices(FIXTURE_ROWS, 0.2, seed=0)
    assert test.tolist() == FIXTURE_TEST_ROWS
    assert train.tolist() == FIXTURE_TRAIN_ROWS


def test_split_is_repeatable_and_depends_on_the_seed():
    first = split_indices(500, 0.2, seed=3)
    again = split_indices(500, 0.2, seed=3)
    other = split_indices(500, 0.2, seed=4)
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])
    assert not np.array_equal(first[1], other[1])


def test_split_does_not_touch_the_global_generator():
    np.random.seed(11)
    expected = np.random.RandomState(11).rand()
    split_indices(100, 0.2, seed=0)
    assert np.random.rand() == expected


@pytest.mark.parametrize("fraction", [0, 1, -0.2, 1.5])
def test_split_rejects_an_impossible_fraction(fraction):
    with pytest.raises(ValueError, match="test fraction must be between 0 and 1"):
        split_indices(100, fraction)


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_split_rejects_too_few_rows(rows):
    with pytest.raises(DataError, match="too few to split"):
        split_indices(rows, 0.2)


def test_split_of_the_smallest_table_it_accepts():
    train, test = split_indices(3, 0.2)
    assert (len(train), len(test)) == (2, 1)


# ----------------------------------------------------------------------------------- the scaler


def test_standardise_fit_on_a_hand_computed_case():
    X = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 40.0]], dtype=np.float32)
    y = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    scaler = standardise_fit(X, y)
    # Column means 3 and 20; population standard deviations sqrt(8 / 3) and sqrt(200).
    assert scaler.x_mean.tolist() == [3.0, 20.0]
    assert scaler.x_std.tolist() == pytest.approx([np.sqrt(8 / 3), np.sqrt(200)], rel=1e-6)
    assert scaler.y_mean == 3.0
    assert scaler.y_std == pytest.approx(np.sqrt(14 / 3), rel=1e-6)
    assert scaler.x_mean.dtype == np.float32
    assert isinstance(scaler.y_mean, float)


def test_standardise_apply_on_a_hand_computed_case():
    scaler = Scaler(
        x_mean=np.array([3.0, 20.0], dtype=np.float32),
        x_std=np.array([2.0, 10.0], dtype=np.float32),
        y_mean=3.0,
        y_std=2.0,
    )
    X, y = standardise_apply(
        np.array([[5.0, 0.0], [3.0, 30.0]], dtype=np.float32),
        np.array([7.0, 2.0], dtype=np.float32),
        scaler,
    )
    assert X.tolist() == [[1.0, -2.0], [0.0, 1.0]]
    assert y.tolist() == [2.0, -0.5]
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_a_constant_column_does_not_divide_by_zero():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)
    y = np.array([4.0, 4.0, 4.0], dtype=np.float32)
    scaler = standardise_fit(X, y)
    X_scaled, y_scaled = standardise_apply(X, y, scaler)
    assert np.all(np.isfinite(X_scaled))
    assert X_scaled[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert y_scaled.tolist() == [0.0, 0.0, 0.0]


def test_destandardise_undoes_the_target_scaling():
    scaler = Scaler(np.zeros(1, np.float32), np.ones(1, np.float32), y_mean=2.0, y_std=0.5)
    assert destandardise_y(np.array([0.0, 2.0, -1.0]), scaler).tolist() == [2.0, 3.0, 1.5]
    y = np.array([0.15, 2.5, 5.00001])
    _, scaled = standardise_apply(np.zeros((3, 1), np.float32), y, scaler)
    assert destandardise_y(scaled.astype(np.float64), scaler) == pytest.approx(y, rel=1e-6)


def test_scalers_match_within_rounding_and_not_across_splits(tmp_path):
    archive = write_archive(tmp_path, random_table(rows=400))
    first = load_california_housing(tmp_path, seed=0, archive=archive).scaler
    again = load_california_housing(tmp_path, seed=0, archive=archive).scaler
    other = load_california_housing(tmp_path, seed=1, archive=archive).scaler
    assert first.matches(again)
    assert not first.matches(other)

    last_bit = Scaler(
        np.nextafter(first.x_mean, np.float32(np.inf)), first.x_std, first.y_mean, first.y_std
    )
    assert first.matches(last_bit)
    shifted = Scaler(first.x_mean, first.x_std, first.y_mean * (1 + 1e-4), first.y_std)
    assert not first.matches(shifted)


# ------------------------------------------------------------------------- the no-leakage rule


def raw_folds(table, seed=0):
    X, y, _ = features_and_target(table)
    train, test = split_indices(len(X), 0.2, seed)
    return X, y, train, test


def test_the_scaling_statistics_are_those_of_the_training_fold_alone(tmp_path):
    table = random_table(rows=200)
    archive = write_archive(tmp_path, table)
    housing = load_california_housing(tmp_path, seed=0, archive=archive)
    X, y, train, test = raw_folds(table)

    # Recomputed here from the training rows only, with no help from the code under test.
    assert np.array_equal(housing.scaler.x_mean, X[train].mean(axis=0))
    assert np.array_equal(housing.scaler.x_std, X[train].std(axis=0) + 1e-7)
    assert housing.scaler.y_mean == float(y[train].mean())
    assert housing.scaler.y_std == float(y[train].std()) + 1e-7

    # They are not the statistics of the whole table, nor of the test fold.
    assert not np.allclose(housing.scaler.x_mean, X.mean(axis=0), rtol=1e-6, atol=0)
    assert not np.allclose(housing.scaler.x_mean, X[test].mean(axis=0), rtol=1e-6, atol=0)
    assert housing.scaler.y_mean != pytest.approx(float(y.mean()), rel=1e-6)


def test_changing_the_test_rows_changes_nothing_the_training_sees(tmp_path):
    table = random_table(rows=200)
    _, _, _, test = raw_folds(table)
    original = load_california_housing(
        tmp_path, seed=0, archive=write_archive(tmp_path, table, "a.tgz")
    )

    # Replace every test row by something wildly different. If any statistic used in training
    # looked at the test fold, the scaler or the training arrays would move.
    altered = table.copy()
    altered[test] = altered[test] * 50 + 1000
    altered[test, 6] = np.abs(altered[test, 6]) + 1
    changed = load_california_housing(
        tmp_path, seed=0, archive=write_archive(tmp_path, altered, "b.tgz")
    )

    assert np.array_equal(changed.scaler.x_mean, original.scaler.x_mean)
    assert np.array_equal(changed.scaler.x_std, original.scaler.x_std)
    assert changed.scaler.y_mean == original.scaler.y_mean
    assert changed.scaler.y_std == original.scaler.y_std
    assert np.array_equal(changed.X_train, original.X_train)
    assert np.array_equal(changed.y_train, original.y_train)
    assert np.array_equal(changed.value_train, original.value_train)
    # The test fold, and only the test fold, is different.
    assert not np.array_equal(changed.X_test, original.X_test)
    assert not np.array_equal(changed.y_test, original.y_test)


def test_changing_a_training_row_does_change_the_statistics(tmp_path):
    table = random_table(rows=200)
    _, _, train, _ = raw_folds(table)
    original = load_california_housing(
        tmp_path, seed=0, archive=write_archive(tmp_path, table, "a.tgz")
    )
    altered = table.copy()
    altered[train[0], 7] += 100
    changed = load_california_housing(
        tmp_path, seed=0, archive=write_archive(tmp_path, altered, "b.tgz")
    )
    assert changed.scaler.x_mean[0] != original.scaler.x_mean[0]
    assert not np.array_equal(changed.X_test, original.X_test)


def test_the_test_fold_is_scaled_with_the_training_statistics(tmp_path):
    table = random_table(rows=200)
    archive = write_archive(tmp_path, table)
    housing = load_california_housing(tmp_path, seed=0, archive=archive)
    X, y, train, test = raw_folds(table)
    mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-7
    y_mean, y_std = float(y[train].mean()), float(y[train].std()) + 1e-7

    assert np.array_equal(housing.X_test, ((X[test] - mean) / std).astype(np.float32))
    assert np.array_equal(housing.y_test, ((y[test] - y_mean) / y_std).astype(np.float32))
    assert np.array_equal(housing.X_train, ((X[train] - mean) / std).astype(np.float32))

    # The training fold is centred and of unit spread by construction; the test fold is only
    # close to it, which is the visible trace of not having fitted on it.
    assert np.allclose(housing.X_train.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(housing.X_train.std(axis=0), 1, atol=1e-4)
    assert np.allclose(housing.y_train.mean(), 0, atol=1e-5)
    assert not np.allclose(housing.X_test.mean(axis=0), 0, atol=1e-4)
    assert not np.allclose(housing.X_test.std(axis=0), 1, atol=1e-4)


def test_no_row_is_in_both_folds(tmp_path):
    table = random_table(rows=200)
    table[:, 8] = 15_000 + np.arange(200)
    archive = write_archive(tmp_path, table)
    housing = load_california_housing(tmp_path, seed=0, archive=archive)
    assert len(housing.value_train) == 160
    assert len(housing.value_test) == 40
    assert len(np.intersect1d(housing.value_train, housing.value_test)) == 0
    assert np.array_equal(
        np.sort(np.concatenate([housing.value_train, housing.value_test])), table[:, 8]
    )


def test_standardise_apply_fits_nothing():
    scaler = Scaler(np.zeros(2, np.float32), np.ones(2, np.float32), y_mean=0.0, y_std=1.0)
    X = np.array([[100.0, -100.0], [300.0, 100.0]], dtype=np.float32)
    y = np.array([50.0, 70.0], dtype=np.float32)
    X_scaled, y_scaled = standardise_apply(X, y, scaler)
    # An identity scaler leaves the fold as it is: nothing was re-centred on the fold itself.
    assert np.array_equal(X_scaled, X)
    assert np.array_equal(y_scaled, y)


# ----------------------------------------------------------------------- load_california_housing


def test_load_returns_the_folds_the_original_loader_produced():
    housing = load_california_housing(FIXTURE_DIR, seed=0, archive=FIXTURE_ARCHIVE)
    assert housing.X_train.shape == (FIXTURE_TRAIN, 8)
    assert housing.y_train.shape == (FIXTURE_TRAIN,)
    assert housing.X_test.shape == (FIXTURE_TEST, 8)
    assert housing.y_test.shape == (FIXTURE_TEST,)
    for array in (housing.X_train, housing.y_train, housing.X_test, housing.y_test):
        assert array.dtype == np.float32
    assert housing.seed == 0

    assert housing.scaler.x_mean.tolist() == pytest.approx(RECORDED_X_MEAN, rel=1e-6)
    assert housing.scaler.x_std.tolist() == pytest.approx(RECORDED_X_STD, rel=1e-6)
    assert housing.scaler.y_mean == pytest.approx(RECORDED_Y_MEAN, rel=1e-6)
    assert housing.scaler.y_std == pytest.approx(RECORDED_Y_STD, rel=1e-6)
    assert housing.X_train[0].tolist() == pytest.approx(RECORDED_X_TRAIN_ROW, rel=1e-4)
    assert housing.y_train[:3].tolist() == pytest.approx(RECORDED_Y_TRAIN, rel=1e-5)
    assert housing.X_test[0].tolist() == pytest.approx(RECORDED_X_TEST_ROW, rel=1e-4)
    assert housing.y_test[:3].tolist() == pytest.approx(RECORDED_Y_TEST, rel=1e-5)


def test_load_keeps_the_published_dollar_values_beside_each_fold():
    housing = load_california_housing(FIXTURE_DIR, seed=0, archive=FIXTURE_ARCHIVE)
    assert housing.value_test.tolist() == [fixture_row(k)[8] for k in FIXTURE_TEST_ROWS]
    assert housing.value_train.tolist() == [fixture_row(k)[8] for k in FIXTURE_TRAIN_ROWS]
    assert housing.value_train.dtype == np.float64
    # Rows 36 to 39 sit at the census cap; the seed 0 split puts all four in the training fold.
    assert int(np.sum(housing.value_train == 500_001)) == 4
    assert int(np.sum(housing.value_test == 500_001)) == 0
    # The standardised target is the dollar value scaled: undoing the scaling gives it back.
    restored = destandardise_y(housing.y_test.astype(np.float64), housing.scaler) * 100_000
    assert restored == pytest.approx(housing.value_test, rel=1e-6)


def test_load_uses_the_module_record_by_default(as_housing):
    housing = load_california_housing(as_housing)
    assert len(housing.X_train) == FIXTURE_TRAIN
    assert load_table(as_housing).shape == (FIXTURE_ROWS, 9)


def test_load_with_another_seed_gives_another_split():
    first = load_california_housing(FIXTURE_DIR, seed=0, archive=FIXTURE_ARCHIVE)
    other = load_california_housing(FIXTURE_DIR, seed=1, archive=FIXTURE_ARCHIVE)
    assert other.seed == 1
    assert not np.array_equal(first.value_test, other.value_test)
    assert not first.scaler.matches(other.scaler)


def test_load_reports_a_damaged_table_with_its_place(tmp_path):
    path = tmp_path / "cal_housing.tgz"
    path.write_bytes(archive_bytes({MEMBER: b"1,2,3\n"}))
    archive = DatasetFile("cal_housing.tgz", path.stat().st_size, sha256_of(path))
    with pytest.raises(DataError, match=r"cal_housing\.data in .*cal_housing\.tgz has 3 columns"):
        load_california_housing(tmp_path, archive=archive)


def test_load_rejects_a_table_too_small_to_split(tmp_path):
    archive = write_archive(tmp_path, random_table(rows=2))
    with pytest.raises(DataError, match="too few to split"):
        load_california_housing(tmp_path, archive=archive)


def test_table_bytes_round_trip():
    table = random_table(rows=5)
    assert np.array_equal(parse_table(table_bytes(table)), table)

"""Tests for spatially blocked cross-validation (geepers.cross_validation)."""

from __future__ import annotations

import numpy as np
import pytest

from geepers.collocation import ordinary_kriging, signal_covariance
from geepers.cross_validation import block_kfold, cross_validate, spatial_blocks


@pytest.fixture
def network():
    rng = np.random.default_rng(12)
    n = 200
    return rng.uniform(-120, -114, n), rng.uniform(33, 39, n)


class TestSpatialBlocks:
    def test_block_size(self, network):
        lon, lat = network
        labels = spatial_blocks(lon, lat, block_km=100)
        # ~550 x 670 km network: a few dozen 100 km blocks, all numbered
        assert 20 < labels.max() + 1 < 70
        assert set(labels) == set(range(labels.max() + 1))

    def test_nearby_points_share_a_block(self):
        lon = np.array([-118.0, -118.001, -112.0])
        lat = np.array([35.0, 35.001, 38.0])
        labels = spatial_blocks(lon + 0.3, lat + 0.3, block_km=50)
        assert labels[0] == labels[1] != labels[2]

    def test_blocks_scale_with_size(self, network):
        lon, lat = network
        small = spatial_blocks(lon, lat, block_km=50).max()
        large = spatial_blocks(lon, lat, block_km=200).max()
        assert small > large


class TestBlockKFold:
    def test_every_point_tested_once(self, network):
        lon, lat = network
        folds = block_kfold(lon, lat, block_km=100, n_splits=5)
        tested = np.concatenate([test for _, test in folds])
        assert sorted(tested) == list(range(lon.size))
        for train, test in folds:
            assert not set(train) & set(test)
            assert len(train) + len(test) == lon.size

    def test_blocks_are_not_split(self, network):
        lon, lat = network
        labels = spatial_blocks(lon, lat, block_km=100)
        for train, test in block_kfold(lon, lat, block_km=100, n_splits=5):
            assert not set(labels[train]) & set(labels[test])

    def test_folds_balanced_on_clustered_network(self):
        rng = np.random.default_rng(3)
        # One dense cluster plus a sparse background
        lon = np.r_[rng.normal(-118, 0.1, 150), rng.uniform(-121, -114, 150)]
        lat = np.r_[rng.normal(35, 0.1, 150), rng.uniform(33, 39, 150)]
        folds = block_kfold(lon, lat, block_km=40, n_splits=4)
        sizes = np.sort([len(test) for _, test in folds])
        # The cluster's block cannot be split, so it sets the largest fold;
        # the remaining points are shared evenly among the other folds
        largest_block = np.bincount(spatial_blocks(lon, lat, block_km=40)).max()
        assert sizes[-1] == largest_block
        assert sizes[-2] - sizes[0] <= 1

    def test_reproducible_and_seed_dependent(self, network):
        lon, lat = network
        a = block_kfold(lon, lat, block_km=100, seed=1)
        b = block_kfold(lon, lat, block_km=100, seed=1)
        c = block_kfold(lon, lat, block_km=100, seed=2)
        assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b, strict=True))
        assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, c, strict=True))

    def test_none_is_plain_kfold(self, network):
        lon, lat = network
        folds = block_kfold(lon, lat, block_km=None, n_splits=5)
        assert [len(test) for _, test in folds] == [40] * 5

    def test_too_few_blocks_raises(self, network):
        lon, lat = network
        with pytest.raises(ValueError, match="occupied blocks"):
            block_kfold(lon, lat, block_km=5000, n_splits=5)


class TestCrossValidate:
    def test_known_predictor(self, network):
        lon, lat = network
        rng = np.random.default_rng(0)
        values = rng.normal(0, 2, lon.size)

        cv = cross_validate(
            lambda _train, test: np.zeros(len(test)), lon, lat, values, block_km=100
        )
        assert cv.rmse == pytest.approx(np.sqrt(np.mean(values**2)))
        np.testing.assert_allclose(cv.residuals, values)
        assert cv.predictions.shape == values.shape
        assert cv.fold_rmse.shape == (5, 1)

    def test_predictor_never_sees_test_points(self, network):
        lon, lat = network
        values = np.arange(lon.size, dtype=float)

        def predict(train, test):
            assert not set(train) & set(test)
            return np.full(len(test), values[train].mean())

        cross_validate(predict, lon, lat, values, block_km=100)

    def test_two_components(self, network):
        lon, lat = network
        rng = np.random.default_rng(1)
        values = np.c_[rng.normal(0, 1, lon.size), rng.normal(0, 5, lon.size)]
        cv = cross_validate(
            lambda _train, test: np.zeros((len(test), 2)),
            lon,
            lat,
            values,
            block_km=100,
        )
        assert cv.rmse.shape == (2,)
        assert cv.rmse[1] > 3 * cv.rmse[0]
        assert cv.residuals.shape == (lon.size, 2)

    def test_mad_resists_an_outlier(self, network):
        lon, lat = network
        rng = np.random.default_rng(2)
        values = rng.normal(0, 1, lon.size)
        values[0] = 500.0
        cv = cross_validate(
            lambda _train, test: np.zeros(len(test)), lon, lat, values, block_km=100
        )
        assert cv.mad[0] == pytest.approx(1.0, abs=0.25)
        assert cv.rmse[0] > 20

    def test_blocking_exposes_optimistic_random_score(self, network):
        # On a spatially correlated field, random hold-out leaves a close
        # neighbor of every test station in training, so it reports a
        # smaller error than predicting across a block-sized gap does
        lon, lat = network
        rng = np.random.default_rng(4)
        parameters = np.array([9.0, 80.0])
        C = signal_covariance(lon, lat, lon, lat, parameters, components=("up",))
        values = np.linalg.cholesky(C + 1e-9 * np.eye(lon.size)) @ rng.normal(
            size=lon.size
        )
        sigmas = np.full(lon.size, 0.1)

        def predict(train, test):
            out = ordinary_kriging(
                lon[train],
                lat[train],
                values[train],
                sigmas[train],
                lon[test],
                lat[test],
                parameters,
            )
            return out.signal[:, 0]

        random = cross_validate(predict, lon, lat, values, block_km=None)
        blocked = cross_validate(predict, lon, lat, values, block_km=150)
        assert blocked.rmse[0] > 1.3 * random.rmse[0]
        # Both still beat predicting the mean
        assert blocked.rmse[0] < values.std()

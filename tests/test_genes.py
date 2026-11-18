"""Tests for evc3d.genes: gene constructor validation."""

import pytest

from evc3d.genes import ConnectionGene


def test_connection_rejects_invalid_surface_attachment():
    with pytest.raises(ValueError, match="Unknown parent face"):
        ConnectionGene(child="child", axis=(0, 1, 0), parent_face="front")

    with pytest.raises(ValueError, match="between -1 and 1"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            surface_uv=(1.1, 0.0),
        )

    with pytest.raises(ValueError, match="child_surface_uv must contain exactly two"):
        ConnectionGene(child="child", axis=(0, 1, 0), child_surface_uv=(0.0,))

    with pytest.raises(ValueError, match="child_surface_uv coordinates must be finite"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            child_surface_uv=(0.0, float("nan")),
        )

    with pytest.raises(ValueError, match="child_surface_uv coordinates must be between"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            child_surface_uv=(0.0, -1.1),
        )


def test_connection_rejects_invalid_child_orientation():
    with pytest.raises(ValueError, match="orientation must contain exactly three"):
        ConnectionGene(child="child", axis=(0, 1, 0), orientation=(0.0, 0.0))

    with pytest.raises(ValueError, match="orientation angles must be finite"):
        ConnectionGene(child="child", axis=(0, 1, 0), orientation=(0.0, float("nan"), 0.0))

    with pytest.raises(ValueError, match="within 90 degrees"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            parent_face="+y",
            orientation=(0.0, 0.0, 0.0),
        )

    with pytest.raises(ValueError, match="within 90 degrees"):
        ConnectionGene(
            child="child",
            axis=(0, 1, 0),
            parent_face="+x",
            orientation=(0.0, 0.0, 90.0),
        )

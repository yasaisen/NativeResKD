"""Public NativeResKD 4.01 configuration facade."""

from __future__ import annotations

from types import SimpleNamespace

from ._NRKD_baseConfig_impl import PROJECT_MAPPING_DICT
from ._NRKD_baseConfig_impl import NRKD_baseConfig as _NRKDBaseConfig


class NRKD_baseConfig(_NRKDBaseConfig):
    """Flat config with environment-correct QC exclusions and legacy views."""

    def setup_datasetHandler(self, project) -> None:
        super().setup_datasetHandler(project)
        self.forbidden_roots = [
            "/work/misaka13/datasets/BRACS"
            if self.is_HPC
            else "/media/yasaisen/NAS8/for_research/datasets/BRACS"
        ]

    @property
    def checkpoint(self):
        return SimpleNamespace(
            save_every_n_epochs=self.save_freq,
            keep_last=True,
            selection_metric=self.checkpoint_selection_metric,
        )


__all__ = ["NRKD_baseConfig", "PROJECT_MAPPING_DICT"]

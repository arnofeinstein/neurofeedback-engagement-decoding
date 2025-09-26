# neuron_analysis/__init__.py  — léger et sans dépendances lourdes

# N'exporte que les utilitaires légers. PAS d'import de decoding/visualization ici.
from .data_loading import get_neuron_files_for_session, build_spike_matrix_from_files

__all__ = [
    "get_neuron_files_for_session",
    "build_spike_matrix_from_files",
    # On laisse decoding/visualization à importer explicitement:
    #   from neuron_analysis.decoding import ...
    #   from neuron_analysis.visualization import ...
    # Et la partie stabilité:
    #   from neuron_analysis.stability import ...
]

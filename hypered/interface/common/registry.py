"""Register interface symbols.

This module provides functionality to register and retrieve symbols (e.g., classes, functions) 
for interface purposes. It maintains an internal dictionary of exported symbols and provides 
decorators and base classes to facilitate the export process.
"""

_EXPORTED: dict[str, type] = {}


def get_symbols():
    """
    Retrieve the dictionary of exported symbols.

    Returns:
        dict[str, type]: A dictionary where the keys are the names of the exported symbols and the values are the symbols themselves.
    """
    return _EXPORTED


def export(symbol):
    """
    Register a symbol for export.

    Args:
        symbol (type): The symbol (e.g., class, function) to register for export.

    Returns:
        type: The registered symbol.
    """
    _EXPORTED[symbol.__name__] = symbol
    return symbol


class exportable:
    """
    Base class for automatically exporting subclasses.

    Any class that inherits from this base class will be automatically registered for export.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register the subclass for export upon its creation.

        Args:
            **kwargs: Additional keyword arguments (currently unused).
        """
        export(cls)

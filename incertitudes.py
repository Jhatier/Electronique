import numpy as np


def incvoltsix(values: np.ndarray) -> np.ndarray:
    """
    Tension DC multimètre 6 et demi

    Fonction créée par Félix et modifiée par mwa
    """
    values = abs(values)

    incertitudes = np.zeros_like(values, dtype=float)
    inc01 = values <= 0.1
    incertitudes[inc01] = 0.00003 * values[inc01] + 0.00003 * 0.1  
    inc1 = (values > 0.1) & (values <= 1)
    incertitudes[inc1] = 0.00002 * values[inc1] + 0.000006 * 1  
    inc10 = (values > 1) & (values <= 10)
    incertitudes[inc10] = 0.000015 * values[inc10] + 0.000004 * 10 
    inc100 = (values > 10) & (values <= 100)
    incertitudes[inc100] = 0.00002 * values[inc100] + 0.000006 * 100
    inc1000 = (values > 100) & (values <= 1000)
    incertitudes[inc1000] = 0.00002 * values[inc1000] + 0.000006 * 1000

    return incertitudes


def incampsix(values: np.ndarray) -> np.ndarray:
    """
    Courant DC multimètre 6 et demi

    Fonction inspirée par celle de Félix
    """
    values = abs(values)

    incertitudes = np.zeros_like(values, dtype=float)
    inc01 = values <= 0.01
    incertitudes[inc01] = 0.00005 * values[inc01] + 0.0001 * 0.01
    inc1 = (values > 0.01) & (values <= 0.1)
    incertitudes[inc1] = 0.0001 * values[inc1] + 0.00004 * 0.1
    inc10 = (values > 0.1) & (values <= 1)
    incertitudes[inc10] = 0.0005 * values[inc10] + 0.00006 * 1
    inc100 = (values > 1) & (values <= 3)
    incertitudes[inc100] = 0.001 * values[inc100] + 0.0002 * 3

    return incertitudes


def incvoltquatre(values: np.ndarray) -> np.ndarray:
    """
    Tension DC multimètre 4 et demi

    Fonction inspirée par celle de Félix
    """
    values = abs(values)

    incertitudes = np.zeros_like(values, dtype=float)
    inc01 = values <= 0.5
    incertitudes[inc01] = 0.0002 * values[inc01] + 0.00001 * 4
    inc1 = (values > 0.5) & (values <= 5)
    incertitudes[inc1] = 0.0002 * values[inc1] + 0.0001 * 4
    inc10 = (values > 5) & (values <= 50)
    incertitudes[inc10] = 0.0002 * values[inc10] + 0.001 * 4
    inc100 = (values > 50) & (values <= 500) 
    incertitudes[inc100] = 0.0002 * values[inc100] + 0.01 * 4
    inc1000 = (values > 500) & (values <= 1200) 
    incertitudes[inc1000] = 0.0002 * values[inc1000] + 0.1 * 4

    return incertitudes

def incampquatre(values: np.ndarray) -> np.ndarray:
    """
    Courant DC multimètre 4 et demi

    Fonction inspirée par celle de Félix
    """
    values = abs(values)

    incertitudes = np.zeros_like(values, dtype=float)
    inc01 = values <= 0.0005
    incertitudes[inc01] = 0.0005 * values[inc01] + 1e-8 * 5
    inc1 = (values > 0.0005) & (values <= 0.005)
    incertitudes[inc1] = 0.0005 * values[inc1] + 1e-7 * 4
    inc10 = (values > 0.005) & (values <= 0.05)
    incertitudes[inc10] = 0.0005 * values[inc10] + 1e-6 * 4
    inc100 = (values > 0.05) & (values <= 0.5)
    incertitudes[inc100] = 0.0005 * values[inc100] + 1e-5 * 4
    inc1000 = (values > 0.5) & (values <= 5)
    incertitudes[inc1000] = 0.0025 * values[inc1000] + 1e-4 * 5
    inc10000 = (values > 5) & (values <= 20)
    incertitudes[inc10000] = 0.0025 * values[inc10000] + 0.001 * 5

    return incertitudes

import time

def get_format(process: str) -> str:
    """Función ficticia para generar enlaces aleatorios"""

    enlaces_procesos_universitarios = {
        "cancelar_inscripcion": "https://www.ejemplo.edu/cancelar-inscripcion",
        "solicitar_transcripcion": "https://www.ejemplo.edu/solicitar-transcripcion",
        "cambiar_carrera": "https://www.ejemplo.edu/cambiar-carrera",
        "pagar_matricula": "https://www.ejemplo.edu/pagar-matricula",
        "solicitar_beca": "https://www.ejemplo.edu/solicitar-beca"
    }

    # Revisar si el proceso está contenido en los índices del diccionario
    process_lower = process.lower()
    for key in enlaces_procesos_universitarios:
        if process_lower in key:
            # Simular demora de llamada a la API
            time.sleep(1)
            return enlaces_procesos_universitarios[key]

    return f"No se encontró ningún enlace para el proceso {process}"
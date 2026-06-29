#Controle de logs (treinamento, debug)

import logging

def get_logger(name="PyMLDA"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(name)
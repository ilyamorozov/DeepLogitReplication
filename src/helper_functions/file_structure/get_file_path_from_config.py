from configparser import ConfigParser
from pathlib import Path

def get_file_path_from_config(path_type: str, path: str):
    """
    curr_dir: directory from which we should get the config.ini (this should refer to the directory in which the top level script lives as that is where the associated config.ini is located)
    path_type: str, refers to the category of path in config.ini from which path should be selected
    path: str, name referencing path to file in config.ini
    """

    config_file = 'src/config.ini'

    config = ConfigParser()
    config.read(config_file)
    file_path = config.get(path_type, path)
    return file_path
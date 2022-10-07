from pathlib import Path
import pathlib


def get_project_root() -> Path:
    '''
    Function to get project root
    '''
    return Path(__file__).parent.parent

def get_projectpaths():
    '''
    Function to get project paths
    '''
    projectroot = get_project_root()
    rawdatapath = pathlib.Path(projectroot / "100_data_raw" / "Australian Obesity Corpus")
    cleandatapath = pathlib.Path(projectroot /  "200_data_clean")
    processeddatapath = pathlib.Path(projectroot / "300_data_processed")
    
    return projectroot, rawdatapath, cleandatapath, processeddatapath
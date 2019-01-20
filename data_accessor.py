# coding: utf8
import logging

from fastai.text import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Change these to your file paths.
DATA_PATH = Path('data/kaggle/toxic_comments')
LM_PATH = Path(DATA_PATH / 'lm/')
CLSFR_PATH = Path(DATA_PATH / 'cls/')

# I placed a copy of fwd_wt103.h5 in this folder:
PRE_LM_PATH = Path('data/pt_models/wt103/')


def get_file_paths():
    """A dictionary of all your file paths."""

    path_dict = {'data_path': DATA_PATH,
                 'lm_path': LM_PATH,
                 'clsfr_path': CLSFR_PATH,
                 'pre_lm_path': PRE_LM_PATH}

    return path_dict


def load_lm_components():
    """Retrieve lm training and validation ids and the itos file."""

    file_paths = get_file_paths()
    lm_path = file_paths['lm_path']

    trn_lm = np.load(lm_path / 'tmp' / 'trn_ids_base.npy')
    val_lm = np.load(lm_path / 'tmp' / 'val_ids_base.npy')
    itos = pickle.load(open(lm_path / 'tmp' / 'itos_base.pkl', 'rb'))
    logger.info('LM training and logging IDs have been loaded from the file systm.')

    return (trn_lm, val_lm, itos)

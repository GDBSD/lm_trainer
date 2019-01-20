# coding: utf8
import datetime
import logging
import time
import torch.tensor as T

from fastai.text import *

import data_accessor as accessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PRE_LM = 'fwd_wt103.h5'
LAST_FIT_MODEL = 'lm_base_last_ft'
TRAINED_MODEL_NAME = 'lm_base_1'
TRAINED_MODEL_ENC = 'lm1_base_1_enc'
ITOS_PKL = 'itos_wt103.pkl'

DROPOUTS = 1


class LmTrainer():
    """You can run this on your GPU instance if training the final version of the backbone
    model will take longer than you want to (or can) keep a Jupyter Notebook open"""

    def train_model(self):
        start = time.time()
        start_time = datetime.datetime.now()
        logger.info('Backbone model training started at {}.'.format(start_time))

        file_paths = accessor.get_file_paths()

        trn_lm, val_lm, itos = accessor.load_lm_components()

        vs = len(itos)

        logger.info('Loaded: vs:{} | trn_lm: {}'.format(vs, len(trn_lm)))

        em_sz, nh, nl = 400, 1150, 3

        pre_lm = '{}/{}'.format(file_paths['pre_lm_path'], PRE_LM)

        wgts = torch.load(pre_lm, map_location=lambda storage, loc: storage)
        logger.info('Loaded the pre-trained model {} from the file system at {}.'.format(
            PRE_LM, file_paths['pre_lm_path']))

        enc_wgts = to_np(wgts['0.encoder.weight'])
        row_m = enc_wgts.mean(0)

        itos2_path = '{}/{}'.format(file_paths['pre_lm_path'], ITOS_PKL)
        itos2 = pickle.load(open(itos2_path, 'rb'))
        stoi2 = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(itos2)})

        new_w = np.zeros((vs, em_sz), dtype=np.float32)
        for i, w in enumerate(itos):
            r = stoi2[w]
            new_w[i] = enc_wgts[r] if r >= 0 else row_m

        wgts['0.encoder.weight'] = T(new_w)
        wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
        wgts['1.decoder.weight'] = T(np.copy(new_w))

        wd = 1e-7
        bptt = 70
        bs = 52
        opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

        trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
        val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)

        # LanguageModelData was refactored into TextLMDataBunch in the new version. Test this to see if
        # it still works if we use TextLMDataBunch here.
        md = LanguageModelData(file_paths['data_path'],
                               1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

        drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * DROPOUTS

        learner = md.get_model(opt_fn, em_sz, nh, nl,
                               dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3],
                               dropouth=drops[4])

        learner.metrics = [accuracy]

        learner.model.load_state_dict(wgts)

        lr = 1e-3
        lrs = lr

        learner.load(LAST_FIT_MODEL)
        logger.info('Loaded the partially trained {} model.'.format(LAST_FIT_MODEL))

        learner.unfreeze()

        training_start_time = datetime.datetime.now()
        logger.info('Started training model at {}.'.format(training_start_time))

        learner.fit(lrs, 1, wds=wd, use_clr=(20, 10), cycle_len=10)

        learner.save(TRAINED_MODEL_NAME)

        learner.save_encoder(TRAINED_MODEL_ENC)

        end = time.time()
        training_time = (end - start)

        logger.info('The {} model has been trained and saved to the file system. Training time was: {}'.format(
            TRAINED_MODEL_NAME, training_time))


if __name__ == '__main__':
    trainer = LmTrainer()
    trainer.train_model()

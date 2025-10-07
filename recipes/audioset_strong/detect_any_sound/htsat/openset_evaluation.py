import argparse
import logging
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

root = "/home/cpf/code/SSL_SED"
os.chdir(root)
sys.path.append(root)

from recipes.audioset_strong.setting import *
from recipes.desed.detect_any_sound.detect_any_sound.finetune.setting import get_logger
from recipes.audioset_strong.detect_any_sound.htsat.train import DASM_HTSAT_Trainer
from src.codec.encoder import Encoder
from src.codec.decoder import batched_decode_preds
from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
from src.utils.statistics.model_statistic import count_parameters
from src.utils import load_yaml_with_relative_ref


def prepare_run():
    torch.backends.cudnn.benchmark = True
    # parse the argument
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='selection of gpu when you run separate trainings on single server')
    parser.add_argument('--config_dir', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--openset_label', type=str, help="path of openset's extra label, saved in json format")
    parser.add_argument('--openset_embedding', type=str, help="path of openset's embedding, saved in pt format")
    parser.add_argument(
        '--common_only',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="True if common classes are base classes only",
    )
    args = parser.parse_args()

    #set configurations
    configs = load_yaml_with_relative_ref(args.config_dir)
    print("=" * 50 + "start!!!!" + "=" * 50)

    configs = get_save_directories(configs, args.save_folder)
    # set logger
    my_logger = get_logger(
        configs["generals"]["save_folder"],
        False,
        log_level=eval("logging." + configs["generals"]["log_level"].upper()),
    )

    my_logger.logger.info("torch version is: " + str(torch.__version__))

    # set device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    configs["training"]["device"] = device
    my_logger.logger.info("device: " + str(device))

    return configs, my_logger, args


def remove_absent_label(config, label_list):
    obj_tsv_path = os.path.join(config["generals"]["save_folder"], '.test.tsv')
    obj_dur_path = os.path.join(config["generals"]["save_folder"], '.dur.tsv')
    # load files
    test_df = pd.read_csv(config["dataset"]["openset_tsv"], sep='\t')
    test_dur_df = pd.read_csv(config["dataset"]["openset_dur"], sep='\t')
    test_df = test_df.loc[test_df['event_label'].isin(label_list)]
    test_dur_df = test_dur_df.loc[test_dur_df['filename'].isin(test_df['filename'].unique())]
    # save files
    test_df.to_csv(obj_tsv_path, sep='\t', index=False, float_format='%.3f')
    test_dur_df.to_csv(obj_dur_path, sep='\t', index=False, float_format='%.3f')
    config["dataset"]["openset_tsv"] = obj_tsv_path
    config["dataset"]["openset_dur"] = obj_dur_path


class Openset_Maskformer_HTSAT_Evaluator(DASM_HTSAT_Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def openset_eval(self, query: torch.Tensor, att_mask: torch.Tensor, openset_label_list):
        self.net.eval()
        n_test = len(self.test_loader)
        # score buffer
        score_buffer = dict()
        # filter type for post-processing
        filter_type = self.config["training"].get("filter_type", "median")
        query = query.to(self.device)
        att_mask = att_mask.to(self.device)
        with torch.no_grad():
            tk = tqdm(self.test_loader, total=n_test, leave=False, desc="test processing")
            for batch in tk:
                wav, labels, pad_mask, idx, filename, path = batch
                wav, labels = wav.to(self.device), labels.to(self.device)
                feat = self.preprocess_eval(wav)
                # prediction for student
                pred = self.net(
                    input=feat,
                    pad_mask=pad_mask,
                    query=query.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1),
                    tgt_mask=att_mask,
                    **self.config[self.net.get_model_name()]["test_kwargs"],
                )
                pred = self.Pred(*pred)

                # =========== update psds score buffer ===========
                raw_scores, scores = batched_decode_preds(
                    strong_preds=pred.strong,
                    filenames=path,
                    encoder=self.encoder,
                    filter=self.median_fiter,
                    weak_preds=None,
                    need_weak_mask=False,
                    filter_type=filter_type,
                )

                score_buffer.update(scores)

        # calculate psds
        test_tsv = self.config["dataset"]["openset_tsv"]
        test_dur = self.config["dataset"]["openset_dur"]
        psds_folders = self.config["training"]["psds_folder"]
        test_df = pd.read_csv(self.config["dataset"]["openset_tsv"], sep='\t')
        test_events_set = set(test_df["event_label"])
        psds, psds_single = self.psds(
            score_buffer,
            test_tsv,
            test_dur,
            save_dir=psds_folders,
            events_set=test_events_set,
        )
        psds_extra = np.mean(np.array([v for k, v in psds_single.items() if k in openset_label_list]))
        # logging
        log_dict = OrderedDict([
            ("psds", psds),
            ("psds extra", psds_extra),
        ])
        # save single psds scores
        psds_single = {k: round(v, 4) for k, v in psds_single.items() if k in openset_label_list}
        with open(os.path.join(self.config["generals"]["save_folder"], 'open_psds.json'), 'w') as f:
            json.dump(psds_single, f)
        self.test_log(log_dict)
        return


if __name__ == "__main__":
    configs, my_logger, args = prepare_run()

    # set network
    net = DASM_HTSAT(**configs["Maskformer_HTSAT"]["init_kwargs"])

    # class label dictionary
    with open(configs['dataset']['label_dict_path']) as f:
        label_dict = json.load(f, object_pairs_hook=OrderedDict)
        label_list = sorted(list(label_dict.keys()))

    # set encoder (with labels for new events in the open-set dataset)
    with open(args.openset_label) as f:
        openset_label = json.load(f)
    with open(configs['dataset']['event_state']) as f:
        type_dict = json.load(f)
    if args.common_only:
        label_list = sorted([label for label in label_list if type_dict[label] == 'common'])

    label_list = label_list + openset_label
    logging.info("<INFO> length of event label list  is {}".format(len(label_list)))
    if args.common_only:
        remove_absent_label(configs, label_list)
    encoder = Encoder(
        label_list,
        audio_len=configs["feature"]["audio_max_len"],
        frame_len=configs["feature"]["win_length"],
        frame_hop=configs["feature"]["hopsize"],
        net_pooling=configs["feature"]["net_subsample"],
        sr=configs["feature"]["sr"],
    )
    # set text query for the detect_any_sound
    query = net.at_query
    if args.common_only:
        common_type_mask = torch.zeros(query.shape[0], dtype=torch.bool)
        for label, idx in label_dict.items():
            if type_dict[label] == 'common':
                common_type_mask[idx] = True
        query = query[common_type_mask, :]

    extra_query = torch.load(args.openset_embedding)
    query = torch.cat((query, extra_query))
    logging.info("<INFO> shape of query  is {}".format(query.shape))

    # set attention mask
    base_size = len(net.at_query) if not args.common_only else torch.count_nonzero(common_type_mask)
    novel_size = extra_query.shape[0]
    att_mask = torch.ones(base_size + novel_size, base_size + novel_size, dtype=torch.bool)
    att_mask[:, :base_size] = False
    att_mask.fill_diagonal_(False)  # fill diagonal in place

    # set dataloader
    devtest_df = pd.read_csv(configs["dataset"]["openset_tsv"], sep="\t")
    test_dataset = StronglyLabeledDataset(
        tsv_read=devtest_df,
        dataset_dir=configs["dataset"]["openset_folder"],
        return_name=True,
        encoder=encoder,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs["training"]["batch_size_val"],
        num_workers=configs["training"]["num_workers"],
    )

    my_logger.logger.info("Total  Params: %.3f M" % (count_parameters(net, trainable_only=False) * 1e-6))
    my_logger.logger.info("Total Trainable Params: %.3f M" % (count_parameters(net) * 1e-6))

    #### move to gpus ########
    net = net.to(configs["training"]["device"])

    evaluator = Openset_Maskformer_HTSAT_Evaluator(
        optimizer=None,
        my_logger=my_logger,
        net=net,
        config=configs,
        encoder=encoder,
        scheduler=None,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        device=configs["training"]["device"],
    )
    evaluator.net.load_state_dict(torch.load(configs['training']["best_paths"]))

    evaluator.openset_eval(query, att_mask, openset_label)

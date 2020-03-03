import argparse
import datetime
import faulthandler
import os

import chainer
import chainermn
import numpy as np
from chainer.training import extensions
from tensorboardX import SummaryWriter

from commands.interactive_train import open_interactive_prompt
from common.dataset_management.dataset_server import DatasetClient
from common.datasets import scatter_dataset
from common.datasets.fsns_dataset import FSNSDataset
from common.datasets.text_recognition_image_dataset import TextRecognitionImageDataset
from config.recognition_config import parse_config
from evaluation.text_recognition_evaluator import TextRecognitionEvaluatorFunction, TextRecognitionTensorboardEvaluator
from insights.fsns_bbox_plotter import FSNSBBoxPlotter
from insights.tensorboard_gradient_histogram import TensorboardGradientPlotter
from insights.text_recognition_bbox_plotter import TextRecognitionBBoxPlotter
from iterators.curriculum_iterator import CurriculumIterator
from optimizers.radam import RAdam
from text.fsns import FSNSTransformerRecognizer, FSNSLSTMLocalizer, FSNSTransformerLocalizer
from text.lstm_text_localizer import LSTMTextLocalizer
from text.transformer_recognizer import TransformerTextRecognizer
from train_utils.backup import get_import_info
from train_utils.datatypes import Size
from train_utils.logger import Logger
from updaters.fsns_updater import FSNSUpdater
from updaters.transformer_text_updater import TransformerTextRecognitionUpdater

faulthandler.enable()


def load_pretrained_model(model_file, model, be_strict=False):
    with np.load(model_file) as handle:
        chainer.serializers.NpzDeserializer(handle, strict=be_strict).load(model)


def main():
    parser = argparse.ArgumentParser(description="Train a KISS model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("log_name", help="name of log")
    parser.add_argument("-c", "--config", default="config.cfg", help="path to config file to use")
    parser.add_argument("--communicator", dest="communicator_name", choices=("flat", "pure_nccl"), default="pure_nccl", help="which mpi communicator to use")
    parser.add_argument("-g", "--gpu", default="@numpy", help="gpu if to use (-1 means cpu)")
    parser.add_argument("-l", "--log-dir", default='tests', help="path to log dir")
    parser.add_argument("--snapshot-interval", type=int, default=10000, help="number of iterations after which a snapshot will be taken")
    parser.add_argument("--log-interval", type=int, default=100, help="log interval")
    parser.add_argument("--port", type=int, default=1337, help="port that is used by bbox plotter to send predictions on test image")
    parser.add_argument("--rl", dest="resume_localizer", help="path to snapshot that is to be used to resume training of localizer")
    parser.add_argument("--rr", dest="resume_recognizer", help="path to snapshot that us to be used to pre-initialize recognizer")
    parser.add_argument("--num-layers", type=int, default=18, help="Resnet Variant to use")
    parser.add_argument("--no-imgaug", action='store_false', dest='use_imgaug', default=True, help="disable image augmentation with `imgaug`, but use naive image augmentation instead")
    parser.add_argument("--rdr", "--rotation-dropout-ratio", dest="rotation_dropout_ratio", type=float, default=0, help="ratio for dropping rotation params in text localization network")
    parser.add_argument("--save-gradient-information", action='store_true', default=False, help="enable tensorboard gradient plotter")
    parser.add_argument("--dump-graph", action='store_true', default=False, help="dump computational graph to file")
    parser.add_argument("--image-mode", default="RGB", choices=["RGB", "L"], help="mode in which images are to be loaded")
    parser.add_argument("--resume", help="path to logdir from which training shall resume")

    args = parser.parse_args()
    args = parse_config(args.config, args)

    comm = chainermn.create_communicator(communicator_name=args.communicator_name)
    if comm.size > 1:
        args.gpu = comm.intra_rank
    print(args.gpu)

    if args.resume is not None:
        log_dir = os.path.relpath(args.resume)
    else:
        log_dir = os.path.join("logs", args.log_dir, "{}_{}".format(datetime.datetime.now().isoformat(), args.log_name))
    args.log_dir = log_dir

    # set dtype
    chainer.global_config.dtype = 'float32'

    if comm.rank == 0:
        # create log dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    report_keys = ["epoch", "iteration", "loss/localizer/loss"]

    if args.use_memory_manager:
        memory_manager = DatasetClient()
        memory_manager.connect()

        train_kwargs = {"memory_manager": memory_manager, "base_name": "train_file"}
        # recognition_kwargs = {"memory_manager": memory_manager, "base_name": "text_recognition_file"}
        validation_kwargs = {"memory_manager": memory_manager, "base_name": "val_file"}
    else:
        train_kwargs = {"npz_file": args.train_file}
        # recognition_kwargs = {"npz_file": args.text_recognition_file}
        validation_kwargs = {"npz_file": args.val_file}

    if comm.rank == 0:
        train_dataset = FSNSDataset(
            char_map=args.char_map,
            image_size=args.image_size,
            root=os.path.dirname(args.train_file),
            dtype=chainer.get_dtype(),
            use_imgaug=args.use_imgaug,
            transform_probability=0.4,
            keep_aspect_ratio=True,
            image_mode=args.image_mode,
            start_level=2,
            **train_kwargs,
        )

        validation_dataset = FSNSDataset(
            char_map=args.char_map,
            image_size=args.image_size,
            root=os.path.dirname(args.val_file),
            dtype=chainer.get_dtype(),
            transform_probability=0,
            keep_aspect_ratio=True,
            image_mode=args.image_mode,
            jump_to_max_level=True,
            **validation_kwargs,
        )
    else:
        train_dataset, validation_dataset = None, None

    # train_dataset = scatter_dataset(train_dataset, comm)
    # validation_dataset = scatter_dataset(validation_dataset, comm)

    data_iter = CurriculumIterator(train_dataset, args.batch_size, curriculum_shift_intervals=[5, 10, 15, 20])
    validation_iter = chainer.iterators.MultithreadIterator(validation_dataset, args.batch_size, repeat=False)

    localizer = FSNSLSTMLocalizer(
        Size(*args.target_size),
        num_bboxes_to_localize=train_dataset.num_words_per_image,
        num_layers=args.num_layers,
        dropout_ratio=args.rotation_dropout_ratio,
    )
    if args.resume_localizer is not None:
        load_pretrained_model(args.resume_localizer, localizer)

    recognizer = FSNSTransformerRecognizer(
        train_dataset.num_chars_per_word,
        train_dataset.num_words_per_image,
        train_dataset.num_classes,
        train_dataset.bos_token,
        num_layers=args.num_layers,
        transformer_size=2048,
    )

    if args.resume_recognizer is not None:
        load_pretrained_model(args.resume_recognizer, recognizer)

    models = [localizer, recognizer]

    if comm.rank == 0:
        tensorboard_handle = SummaryWriter(log_dir=args.log_dir)
    else:
        tensorboard_handle = None

    localizer_optimizer = RAdam(alpha=args.learning_rate, beta1=0.9, beta2=0.98, eps=1e-9)
    localizer_optimizer = chainermn.create_multi_node_optimizer(localizer_optimizer, comm)
    localizer_optimizer.setup(localizer)
    localizer_optimizer.add_hook(
        chainer.optimizer_hooks.GradientClipping(2)
    )

    if args.save_gradient_information:
        localizer_optimizer.add_hook(
            TensorboardGradientPlotter(tensorboard_handle, args.log_interval),
        )

    recognizer_optimizer = RAdam(alpha=args.learning_rate)
    recognizer_optimizer = chainermn.create_multi_node_optimizer(recognizer_optimizer, comm)
    recognizer_optimizer.setup(recognizer)

    optimizers = [localizer_optimizer, recognizer_optimizer]

    # log train information everytime we encouter a new epoch or args.log_interval iterations have been done
    log_interval_trigger = (
        lambda trainer:
        (trainer.updater.is_new_epoch or trainer.updater.iteration % args.log_interval == 0)
        and trainer.updater.iteration > 0
    )

    updater_args = {
        "iterator": {
            'main': data_iter,
        },
        "optimizer": {
            "opt_gen": localizer_optimizer,
            "opt_rec": recognizer_optimizer,
        },
        "tensorboard_handle": tensorboard_handle,
        "tensorboard_log_interval": log_interval_trigger,
        "recognizer_update_interval": 1,
        "device": args.gpu,
    }

    updater = FSNSUpdater(
        models=[localizer, recognizer],
        **updater_args
    )

    trainer = chainer.training.Trainer(updater, (args.num_epoch, 'epoch'), out=args.log_dir)

    data_to_log = {
        'log_dir': args.log_dir,
        'image_size': args.image_size,
        'num_layers': args.num_layers,
        'num_chars': train_dataset.num_chars_per_word,
        'num_words': train_dataset.num_words_per_image,
        'num_classes': train_dataset.num_classes,
        'keep_aspect_ratio': train_dataset.keep_aspect_ratio,
        'bos_token': train_dataset.bos_token,
    }

    for argument in filter(lambda x: not x.startswith('_'), dir(args)):
        data_to_log[argument] = getattr(args, argument)

    if tensorboard_handle is not None:
        tensorboard_handle.add_hparams({k: v for k, v in data_to_log.items() if v is not None}, {})

    data_to_log.update({
        'localizer': get_import_info(localizer),
        'recognizer': get_import_info(recognizer),
    })

    def backup_train_config(stats_cpu):
        if stats_cpu['iteration'] == args.log_interval:
            stats_cpu.update(data_to_log)

    trainer.extend(
        extensions.snapshot(filename='trainer_snapshot', autoload=args.resume is not None),
        trigger=(args.snapshot_interval, 'iteration')
    )

    if comm.rank == 0:
        for model in models:
            trainer.extend(
                extensions.snapshot_object(model, model.__class__.__name__ + '_{.updater.iteration}.npz'),
                trigger=(args.snapshot_interval, 'iteration')
            )

        evaluation_function = TextRecognitionEvaluatorFunction(localizer, recognizer, args.gpu, train_dataset.blank_label, train_dataset.char_map)

        trainer.extend(
            TextRecognitionTensorboardEvaluator(
                validation_iter,
                localizer,
                device=args.gpu,
                eval_func=evaluation_function,
                tensorboard_handle=tensorboard_handle,
                num_iterations=200,
            ),
            trigger=(args.test_interval, 'iteration'),
        )

        # every epoch run the model on test datasets
        test_dataset_prefix = "test_dataset_"
        test_datasets = [arg for arg in dir(args) if arg.startswith(test_dataset_prefix)]
        for test_dataset_name in test_datasets:
            print(f"setting up testing for {test_dataset_name[len(test_dataset_prefix):]} dataset")

            dataset_path = getattr(args, test_dataset_name)
            if args.use_memory_manager:
                test_kwargs = {"memory_manager": memory_manager, "base_name": test_dataset_name}
            else:
                test_kwargs = {"npz_file": dataset_path}

            test_dataset = FSNSDataset(
                char_map=args.char_map,
                image_size=args.image_size,
                root=os.path.dirname(dataset_path),
                dtype=chainer.get_dtype(),
                transform_probability=0,
                keep_aspect_ratio=True,
                image_mode=args.image_mode,
                jump_to_max_level=True,
                **test_kwargs,
            )
            test_iter = chainer.iterators.MultithreadIterator(test_dataset, args.batch_size, repeat=False)
            trainer.extend(
                TextRecognitionTensorboardEvaluator(
                    test_iter,
                    localizer,
                    device=args.gpu,
                    eval_func=evaluation_function,
                    tensorboard_handle=tensorboard_handle,
                    base_key=test_dataset_name[len(test_dataset_prefix):]
                ),
                trigger=(args.snapshot_interval, 'iteration')
            )

        models.append(updater)
        logger = Logger(
            os.path.dirname(os.path.realpath(__file__)),
            args.log_dir,
            postprocess=backup_train_config,
            trigger=log_interval_trigger,
            exclusion_filters=['*logs*', '*.pyc', '__pycache__', '.git*'],
            resume=args.resume is not None,
        )

        if args.test_image is not None:
            plot_image = train_dataset.load_image(args.test_image)
            gt_bbox = None
        else:
            plot_image = validation_dataset.get_example(0)['image']
            gt_bbox = None

        bbox_plotter = FSNSBBoxPlotter(
            plot_image,
            os.path.join(args.log_dir, 'bboxes'),
            args.target_size,
            send_bboxes=True,
            upstream_port=args.port,
            visualization_anchors=[
                ["visual_backprop_anchors"],
            ],
            device=args.gpu,
            render_extracted_rois=True,
            num_rois_to_render=4,
            sort_rois=False,
            show_visual_backprop_overlay=True,
            visual_backprop_index=0,
            show_backprop_and_feature_vis=True,
            gt_bbox=gt_bbox,
            render_pca=False,
            log_name=args.log_name,
            char_map=train_dataset.char_map,
            blank_label=train_dataset.blank_label,
            predictors={
                "localizer": localizer,
                "recognizer": recognizer,
            },
        )
        trainer.extend(bbox_plotter, trigger=(10, 'iteration'))

        trainer.extend(
            logger,
            trigger=log_interval_trigger
        )
        trainer.extend(
            extensions.PrintReport(report_keys, log_report='Logger'),
            trigger=log_interval_trigger
        )

        # learning rate shift after each epoch
        trainer.extend(
            extensions.ExponentialShift("alpha", 0.1, optimizer=localizer_optimizer),
            trigger=(10, 'epoch')
        )

        trainer.extend(extensions.ProgressBar(update_interval=10))

        if args.dump_graph:
            trainer.extend(extensions.dump_graph('loss/localizer/loss', out_name='model.dot'))

        open_interactive_prompt(
            bbox_plotter=bbox_plotter,
            optimizer=optimizers,
        )

    trainer.run()


if __name__ == "__main__":
    main()

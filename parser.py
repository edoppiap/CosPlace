import os
import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152", "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14", "maxvit_t",
                                 "efficientnet_v2_s", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenet_v3_small", "mobilenet_v3_large"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    # Optimizer
    parser.add_argument("--optimizer", type=str, default='AdamW',
                        choices=["AdamW", "Adam", "SGD", "Adagrad", "LBFGS", "Adadelta"],
                        help="Optimizer to use")
    # Domain adaptation parameters & Data augmentation
    parser.add_argument("--domain_adapt", type=str, default=None,
                        help="It turns on Domain Adaptation training")
    # Scheduler
    parser.add_argument('--scheduler', type=str, default=None, 
                        choices=["StepLR","ReduceLROnPlateau","CosineAnnealignLR","ExponentialLR"],
                        help='scheduler to use')    
    #Loss
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', 
                        choices=["CrossEntropyLoss", "VICRegLoss", "TripletMarginLoss"], 
                        help='loss to use')

    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=50, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    # Domain adaptation learning rate
    parser.add_argument("--lr_domain_adapt", type=float, default=0.0001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0, help="_")
    parser.add_argument("--horizontal_flip_prob", type=float, default=0, help="_")
    parser.add_argument("--vertical_flip_prob", type=float, default=0, help="_")
    parser.add_argument("--autoaugment_policy", type=str, default=None,
                        choices=["IMAGENET", "CIFAR10", "SVHN"], nargs="+",
                        help="Policy for AutoAugment augmentations (you can pick more than one)")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Multi scale parameters
    parser.add_argument("--multi_scale", action='store_true', help="Use multi scale")
    parser.add_argument("--select_resolutions", type=float, default=[0.526, 0.588, 1, 1.7, 1.9], nargs="+",
                        help="Usage: --select_resolution 1 2 4 6")
    parser.add_argument("--multi_scale_method", type=str, default="avg", choices=["avg", "sum", "max", "min"],
                        help="Usage:--multi_scale_method=avg")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # GeoWarp parameters
    parser.add_argument("--k", type=int, default=0.6,
                        help="parameter k, defining the difficulty of ss training data")
    parser.add_argument("--ss_w", type=float, default=1,
                        help="weight of self-supervised loss")
    parser.add_argument("--consistency_w", type=float, default=0.1,
                        help="weight of consistency loss")
    parser.add_argument("--features_wise_w", type=float, default=10,
                        help="weight of features-wise loss")
    parser.add_argument("--qp_threshold", type=float, default=1.2,
                        help="Threshold distance (in features space) for query-positive pairs")
    parser.add_argument("--num_reranked_preds", type=int, default=5,
                        help="number of predictions to re-rank at test time")
    parser.add_argument("--kernel_sizes", nargs='+', default=[7, 5, 5, 5, 5, 5],
                        help="size of kernels in conv layers of Homography Regression")
    parser.add_argument("--channels", nargs='+', default=[225, 128, 128, 64, 64, 64, 64],
                        help="num channels in conv layers of Homography Regression")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                             "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                             "preds for difficult queries, i.e. with uncorrect first prediction")
    # Paths parameters
    parser.add_argument("--dataset_folder", type=str, default=None,
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")

    args = parser.parse_args()

    if args.dataset_folder is None:
        try:
            args.dataset_folder = os.environ['SF_XL_PROCESSED_FOLDER']
        except KeyError:
            raise Exception("You should set parameter --dataset_folder or export " +
                            "the SF_XL_PROCESSED_FOLDER environment variable as such \n" +
                            "export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed")

    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")

    if is_training:
        args.train_set_folder = os.path.join(args.dataset_folder, "train")
        if not os.path.exists(args.train_set_folder):
            raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")

        args.val_set_folder = os.path.join(args.dataset_folder, "val")
        if not os.path.exists(args.val_set_folder):
            raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")

    args.test_set_folder = os.path.join(args.dataset_folder, "test")
    if not os.path.exists(args.test_set_folder):
        raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")

    return args
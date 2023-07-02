import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import util
import parser
import commons
import cosface_loss
import augmentations
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
import geowarp
import test_geowarp

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

import geowarp_dataset

# #### Model
# if args.domain_adapt == 'True':
#     model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim,
#                                                 alpha=0.05, domain_adapt="True")
#     logging.info(f"Using domain adaption")
# else:
#     model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, alpha=None, domain_adapt=None)
#     logging.info(f"Using domain adaption")

features_extractor = geowarp.FeatureExtractor(args.backbone, args.fc_output_dim)
global_features_dim = commons.get_output_dim(features_extractor, "gem")

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    features_extractor.load_state_dict(model_state_dict)
    del model_state_dict

homography_regression = geowarp.HomographyRegression(kernel_sizes=args.kernel_sizes, channels=args.channels, padding=1)
model = geowarp.GeoWarp(features_extractor, homography_regression).cuda().eval()
model = torch.nn.DataParallel(model)

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
mse = torch.nn.MSELoss()
# UPDATE: request f. adding or trying with a new optimizer from Adam to AdamW
if args.optimizer == "Adam":
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "AdamW":
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
elif args.optimizer == "SGD":
    model_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == "Adagrad":
    model_optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
elif args.optimizer == "LBFGS":
    model_optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)
elif args.optimizer == "Adadelta":
    model_optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)

### Scheduler
if args.scheduler == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=30, gamma=0.1)
elif args.scheduler == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
elif args.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=50, eta_min=0)
# Add more elif conditions for other schedulers you want to use
elif args.scheduler == 'ExponentialLR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.95)
else:
    print("Invalid scheduler choice")

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
# UPDATE: request f. adding or trying with a new optimizer from Adam to AdamW
# classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]
classifiers_optimizers = [torch.optim.AdamW(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                          classifiers]
ss_dataset = [geowarp_dataset.HomographyDataset(args, args.train_set_folder, M=args.M, N=args.N, current_group=n, min_images_per_class=args.min_images_per_class, k=args.k) for n in range(args.groups_num)] # k = parameter k, defining the difficulty of ss training data, default = 0.6

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(
    f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

# Dataset day label (1,1,1)
groups_day = [TrainDataset(args, "/content/data/tokyo_xs/day_database/", M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                           current_group=n, min_images_per_class=args.min_images_per_class, day=True) for n in
              range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers_day = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups_day]
classifiers_optimizers_day = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                              classifiers_day]

# How many classes and images for the day domain label prediction
logging.info(f"Using {len(groups_day)} groups")
logging.info(
    f"The {len(groups_day)} groups have respectively the following number of classes {[len(g) for g in groups_day]}")
logging.info(
    f"The {len(groups_day)} groups have respectively the following number of images {[g.get_images_num() for g in groups_day]}")

logging.info(f"Day sunny trial group: {groups_day[0]} ")

# Dataset night label (0,0,0)
# path kaggle:  "/kaggle/working/data/tokyo_xs/night"
# path to pc:
groups_night = [TrainDataset(args, "/content/data/tokyo_xs/night_database/database",
                             M=args.M, alpha=args.alpha, N=args.N, L=args.L, current_group=n,
                             min_images_per_class=args.min_images_per_class, night=True) \
                for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group1
classifiers_night = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups_night]
classifiers_optimizers_night = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                                classifiers_night]

# How many classes and images for the night domain label prediction
logging.info(f"Using {len(groups_night)} groups")
logging.info(
    f"The {len(groups_night)} groups have respectively the following number of classes {[len(g) for g in groups_night]}")
logging.info(
    f"The {len(groups_night)} groups have respectively the following number of images {[g.get_images_num() for g in groups_night]}")

logging.info(f"Day night version trial group: {groups_night[0]} ")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

if args.augmentation_device == "cuda":
    compose = []
    compose.append(augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                           contrast=args.contrast,
                                                           saturation=args.saturation,
                                                           hue=args.hue))
    compose.append(augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                                 scale=[1 - args.random_resized_crop, 1]))
    compose.append(augmentations.DeviceAgnosticRandomHorizontalFlip(args.horizontal_flip_prob))
    compose.append(T.RandomVerticalFlip(args.vertical_flip_prob))
    if args.autoaugment_policy:
        for policy_name in args.autoaugment_policy:  # it can be more than one
            logging.info(f"Selected AutoAugment policy: {policy_name}")
            compose.append(augmentations.DeviceAgnosticAutoAugment(policy_name=policy_name,
                                                                   interpolation=T.InterpolationMode.NEAREST))
    compose.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    gpu_augmentation = T.Compose(compose)

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

# If domain adaption is set to True we train
for epoch_num in range(start_epoch_num, args.epochs_num):
    #### Train
    epoch_start_time = datetime.now()
    current_group_num = epoch_num % args.groups_num


    # Select classifier and dataloader according to epoch
    def select_classifier(classifier, optimizer):
        classifier = classifier.to(args.device)
        util.move_to_device(optimizer, args.device)
        return classifier


    if args.domain_adapt == 'True':
        classifiers[current_group_num] = select_classifier(classifiers[current_group_num],
                                                           classifiers_optimizers[current_group_num])
        classifiers_day[current_group_num] = select_classifier(classifiers_day[current_group_num],
                                                               classifiers_optimizers_day[current_group_num])
        classifiers_night[current_group_num] = select_classifier(classifiers_night[current_group_num],
                                                                 classifiers_optimizers_night[current_group_num])

        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)
        dataloader_day = commons.InfiniteDataLoader(groups_day[current_group_num], num_workers=args.num_workers,
                                                    batch_size=args.batch_size, shuffle=True,
                                                    pin_memory=(args.device == "cuda"), drop_last=True)
        dataloader_night = commons.InfiniteDataLoader(groups_night[current_group_num], num_workers=args.num_workers,
                                                      batch_size=args.batch_size, shuffle=True,
                                                      pin_memory=(args.device == "cuda"), drop_last=True)
        ss_dataloader = commons.InfiniteDataLoader(ss_dataset[current_group_num], num_workers=args.num_workers,
                                                   batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=(args.device == "cuda"), drop_last=True)

        dataloader_iterator = iter(dataloader)
        dataloader_iterator_day = iter(dataloader_day)
        dataloader_iterator_night = iter(dataloader_night)
        ss_iterator = iter(ss_dataloader)

        logging.info(f"Dataloader CLASSIC: {len(dataloader)}")
        logging.info(f"Dataloader DAY: {len(dataloader_day)}")
        logging.info(f"Dataloader NIGHT: {len(dataloader_night)}")

        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = next(ss_iterator)  # dal warping dataset prende le due immagini warped e i due punti delle intersezioni
        warped_img_1, warped_img_2, warped_intersection_points_1, warped_intersection_points_2 = warped_img_1.to(
            args.device), warped_img_2.to(args.device), warped_intersection_points_1.to(
            args.device), warped_intersection_points_2.to(args.device)  # warping dataset

    else:
        classifiers[current_group_num] = select_classifier(classifiers[current_group_num],
                                                           classifiers_optimizers[current_group_num])
        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)
        dataloader_iterator = iter(dataloader)

    model = model.train()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        if args.domain_adapt == 'True':
            images, targets, _ = next(dataloader_iterator)
            # images_day, targets_day, _ = next(dataloader_iterator_day)
            # images_night, targets_night, _ = next(dataloader_iterator_night)

            images, targets = images.to(args.device), targets.to(args.device)
            # images_day, targets_day = images_day.to(args.device), targets_day.to(args.device)
            # images_night, targets_night = images_night.to(args.device), targets_night.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)
                # images_day = gpu_augmentation(images_day)
                # images_night = gpu_augmentation(images_night)
        else:
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(args.device), targets.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)


        with torch.no_grad():
            similarity_matrix_1to2, similarity_matrix_2to1 = model("similarity", [warped_img_1, warped_img_2])

        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        descriptors = model("features_extractor", [images, "global"])
        output = classifiers[current_group_num](descriptors, targets)
        loss = criterion(output, targets)
        loss.backward()
        loss = loss.item()
        del output, images
        # ss_loss
        if args.ss_w != 0:
            pred_warped_intersection_points_1 = model("regression", similarity_matrix_1to2)
            pred_warped_intersection_points_2 = model("regression", similarity_matrix_2to1)
            ss_loss = (mse(pred_warped_intersection_points_1[:, :4], warped_intersection_points_1) +
                       mse(pred_warped_intersection_points_1[:, 4:], warped_intersection_points_2) +
                       mse(pred_warped_intersection_points_2[:, :4], warped_intersection_points_2) +
                       mse(pred_warped_intersection_points_2[:, 4:], warped_intersection_points_1))
            # ss_loss = compute_loss(ss_loss, args.ss_w)
            ss_loss.backward()
            ss_loss = ss_loss.item()
            del pred_warped_intersection_points_1, pred_warped_intersection_points_2
        else:
            ss_loss = 0
        epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss]])))  # both losses
        del loss, ss_loss
        # step update
        model_optimizer.step()
        classifiers_optimizers[current_group_num].step()
    else:  # Use AMP 16
        with torch.cuda.amp.autocast():
            descriptors = model("features_extractor", [images, "global"])
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
        scaler.scale(loss).backward()
        epoch_losses = np.append(epoch_losses, loss.item())
        del loss, output, images
        scaler.step(model_optimizer)
        scaler.step(classifiers_optimizers[current_group_num])
        scaler.update()

        # remember to concatenate both losses
        epoch_losses = np.concatenate((epoch_losses, np.array([[loss, ss_loss]])))  # both losses
        del loss, ss_loss
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")

    ### Evaluation (still in the for - delete the ())
    recalls, recalls_str, _ = test_geowarp.use_geowarp(args, val_ds, model)
    logging.info(
        f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)
    #### end of evaluation
    ### end of training.

logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info("Experiment finished (without any errors)")

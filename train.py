import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
from pytorch_metric_learning import distances, losses, miners, reducers, testers


import test
import util
import parser
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
if args.domain_adapt == 'True':
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim,
                                                alpha=0.05, domain_adapt="True")
    logging.info(f"Using domain adaption")
else:
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.use_places, alpha=None, domain_adapt=None)
    logging.info(f"Using domain adaption")

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

### Loss 
if args.loss == 'CrossEntropyLoss':
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss == 'TripletMarginLoss':
    criterion = losses.SelfSupervisedLoss(losses.TripletMarginLoss(margin=0.05))
elif args.loss == 'VICRegLoss':
    criterion = losses.VICRegLoss(invariance_lambda=25, 
                                variance_mu=25, 
                                covariance_v=1, 
                                eps=1e-4)
    
    
    
#### Optimizer
# 
#UPDATE: request f. adding or trying with a new optimizer from Adam to AdamW

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
elif args.scheduler == 'CosineAnnealignLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=50, eta_min=0)
elif args.scheduler == 'ExponentialLR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.95)
else:
    scheduler = None
    
#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
# UPDATE: request f. adding or trying with a new optimizer from Adam to AdamW
# classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]
classifiers_optimizers = [torch.optim.AdamW(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                          classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(
    f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
test_tokyo_night_ds = TestDataset("/content/data/tokyo_xs/night_database/", queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
test_tokyo_day_ds = TestDataset("/content/data/tokyo_xs/day_database/", queries_folder="queries",
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
    if args.loss == "TripletMarginLoss" or args.loss == 'VICRegLoss':
        compose2 = []
        compose2.append(augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                          scale=[1-.5, 1]))
        compose2.append(augmentations.DeviceAgnosticRandomHorizontalFlip(.5))
        compose2.append(T.RandomVerticalFlip(.5))
        compose2.append(T.RandomErasing(0.5))
        compose2.append(T.RandomPerspective(0.5))
        compose2.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        gpu_augmentation_2 = T.Compose(compose2)
    

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

        dataloader_iterator = iter(dataloader)
        dataloader_iterator_day = iter(dataloader_day)
        dataloader_iterator_night = iter(dataloader_night)

        logging.info(f"Dataloader CLASSIC: {len(dataloader)}")
        logging.info(f"Dataloader DAY: {len(dataloader_day)}")
        logging.info(f"Dataloader NIGHT: {len(dataloader_night)}")

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
            images_day, targets_day, _ = next(dataloader_iterator_day)
            images_night, targets_night, _ = next(dataloader_iterator_night)

            images, targets = images.to(args.device), targets.to(args.device)
            images_day, targets_day = images_day.to(args.device), targets_day.to(args.device)
            images_night, targets_night = images_night.to(args.device), targets_night.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)
                images_day = gpu_augmentation(images_day)
                images_night = gpu_augmentation(images_night)
        else:
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(args.device), targets.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)
                if args.loss == 'TripletMarginLoss':
                  augmented = gpu_augmentation_2(images)   

        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        if not args.use_amp16:
            descriptors = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            if args.loss == 'TripletMarginLoss':
                augmented_descriptors = model(augmented)
                augmented_output = classifiers[current_group_num](augmented_descriptors, targets)
                loss = criterion(descriptors, augmented_descriptors)
            elif args.loss == 'VICRegLoss':
                loss = criterion(output, ref_emb=None)
            else:
                loss = criterion(output, targets)
            loss.backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images

            if args.domain_adapt == 'True':
                descriptors_day = model(images_day, alpha=0.05)
                output_day = classifiers_day[current_group_num](descriptors_day, targets_day)
                loss_day = criterion(output_day, targets_day)

                descriptors_night = model(images_night, alpha=0.05)
                output_night = classifiers_night[current_group_num](descriptors_night, targets_night)
                loss_night = criterion(output_night, targets_night)

                loss_domain = loss_night + loss_day
                loss_domain.backward()

                epoch_losses = np.append(epoch_losses, loss_day.item())
                epoch_losses = np.append(epoch_losses, loss_night.item())
                del loss_day, loss_night, output_day, output_night, images_day, images_night
                classifiers_optimizers_day[current_group_num].step()
                classifiers_optimizers_night[current_group_num].step()

            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()

        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                if args.loss == 'TripletMarginLoss':
                    augmented_descriptors = model(augmented)
                    augmented_output = classifiers[current_group_num](augmented_descriptors, targets)
                    loss = criterion(descriptors, augmented_descriptors)
                elif args.loss == 'VICRegLoss':
                    loss = criterion(output, ref_emb=None)
                else:
                    loss = criterion(output, targets)

                if args.domain_adapt == 'True':
                    descriptors_day = model(images_day)
                    output_day = classifiers_day[current_group_num](descriptors_day, targets_day)
                    loss_day = criterion(output_day, targets_day)

                    descriptors_night = model(images_night)
                    output_night = classifiers_night[current_group_num](descriptors_night, targets_night)
                    loss_night = criterion(output_night, targets_night)

                    loss_domain = loss_night + loss_day

            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images

            if args.domain_adapt == 'True':
                scaler.scale(loss_day).backward()
                scaler.scale(loss_night).backward()

                epoch_losses = np.append(epoch_losses, loss_day.item())
                epoch_losses = np.append(epoch_losses, loss_night.item())
                del loss_day, loss_night, output_day, output_night, images_day, images_night
                scaler.step(classifiers_optimizers_day[current_group_num])
                scaler.step(classifiers_optimizers_night[current_group_num])

            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scale = scaler.get_scale()
            scaler.update()

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_losses.mean())
        else:
            scheduler.step()
    

    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(
        f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)

    # Save checkpoint, which contains all training parameters
    checkpoint = {
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }
    util.save_checkpoint(checkpoint, is_best, args.output_folder)

logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")

# Testing on both Tokyo Night and Tokyo Day
logging.info(f"Now testing on the Tokyo set night: {test_tokyo_night_ds}")
recalls, recalls_str = test.test(args, test_tokyo_night_ds, model, args.num_preds_to_save)
logging.info(f"{test_tokyo_night_ds}: {recalls_str}")

logging.info(f"Now testing on the Tokyo set day: {test_tokyo_day_ds}")
recalls, recalls_str = test.test(args, test_tokyo_day_ds, model, args.num_preds_to_save)
logging.info(f"{test_tokyo_day_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")

import argparse
import os
import time
import logging
import numpy as np
import torch
import torchvision

from src.dataset import RenderedDataset
from src.predictor import SO3Predictor, ClassPredictor

def evaluate_pose_acc(model, test_loader, class_names):
    model.eval()

    accuracies = []
    class_ids = []
    for batch_idx, batch in enumerate(test_loader):
        class_ids.append(batch[1].numpy())
        batch = map(lambda d: d.to('cuda'), batch)
        _, data = model.compute_loss(*batch)
        accuracies.append(data['acc'])

    accuracies = np.concatenate(accuracies).flatten()
    class_ids = np.concatenate(class_ids).flatten()

    eval_data = {}
    med_errs = []
    for cls_id, name in enumerate(class_names):
        mask = class_ids == cls_id
        acc = accuracies[mask]
        eval_data[name] = acc
        med_err = np.degrees(np.median(acc))
        med_errs.append(med_err)
        print(f'{name}: mederr = {med_err:.1f}°')
    print(f'AVERAGE: mederr = {np.mean(med_errs):.1f}°')

    return eval_data


def evaluate_classification(model, test_loader, class_names):
    model.eval()

    accuracies = []
    class_ids = []
    for batch_idx, batch in enumerate(test_loader):
        class_ids.append(batch[1].numpy())
        batch = map(lambda d: d.to('cuda'), batch)
        _, data = model.compute_loss(*batch)
        accuracies.append(data['acc'])

    accuracies = np.concatenate(accuracies).flatten()
    class_ids = np.concatenate(class_ids).flatten()

    eval_data = {}
    for cls_id, name in enumerate(class_names):
        mask = class_ids == cls_id
        acc = accuracies[mask]
        eval_data[name] = acc
        print(f'{name}: Acc = {np.mean(acc):.2f}')
    print(f'AVERAGE: Acc = {np.mean(accuracies):.2f}')

    return eval_data


def create_dataloaders(args):
    train_set = RenderedDataset(dataset_name=args.dataset_name,
                                task=args.task,
                                objects=args.objects,
                                mode='train',
                                img_mode=args.img_mode,
                                shift_aug=args.pixel_shift_aug,
                                depth_aug=args.depth_shift_aug,
                               )

    test_set = RenderedDataset(dataset_name=args.dataset_name,
                                task=args.task,
                                objects=args.objects,
                                mode='test',
                                img_mode=args.img_mode,
                              )

    print(f'Training on {train_set.num_classes} classes: {", ".join(train_set.class_names)}')
    print(f'{len(train_set)} train imgs; {len(test_set)} test imgs')

    args.img_shape = train_set.img_shape
    args.num_classes = train_set.num_classes
    args.class_names = train_set.class_names

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers
                                             )
    return train_loader, test_loader, args


def create_model(args):
    if args.task == 'pose':
        model = SO3Predictor(img_shape=args.img_shape,
                             num_classes=args.num_classes,
                             sphere_fdim=args.sphere_fdim,
                             ico_level=args.ico_level,
                             offset_lambda=args.offset_lambda,
                             encoder_type=args.encoder_type,
                             encoder_kwargs={'order' : args.encoder_order},
                            ).to(args.device)
    elif args.task == 'classification':
        model = ClassPredictor(img_shape=args.img_shape,
                               num_classes=args.num_classes,
                               sphere_fdim=args.sphere_fdim,
                               ico_level=args.ico_level,
                               encoder_type=args.encoder_type,
                               encoder_kwargs={'order' : args.encoder_order},
                              ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params: {num_params/1e6:.3f}M')

    model.train()
    return model


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    objects = '-'.join(args.objects)
    fname = f"{args.dataset_name}_{objects}_{args.img_mode}_{args.encoder_type}_seed{args.seed}"
    args.fdir = os.path.join(args.results_dir, fname)
    print(args.fdir)

    if not os.path.exists(args.fdir):
        os.makedirs(args.fdir)

    with open(os.path.join(args.fdir, 'args.txt'), 'w') as f:
        f.write(str(args.__dict__))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers =  [logging.StreamHandler(),
                        logging.FileHandler(os.path.join(args.fdir, "log.txt"))]
    logger.info("%s", repr(args))

    if os.path.exists(os.path.join(args.fdir, 'data.txt')):
        os.remove(os.path.join(args.fdir, 'data.txt'))

    train_loader, test_loader, args = create_dataloaders(args)

    model = create_model(args)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr_initial,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay,
                                nesterov=bool(args.sgd_nesterov),
                               )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   args.lr_step_size,
                                                   args.lr_decay_rate)


    data = []
    for epoch in range(args.num_epochs):
        train_cls_loss = 0
        train_reg_loss = 0
        train_acc = []
        time_before_epoch = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            batch = map(lambda d: d.to(args.device), batch)
            loss, loss_info = model.compute_loss(*batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_cls_loss += loss_info['cls_loss']
            train_reg_loss += loss_info['reg_loss']
            train_acc.append(loss_info['acc'])

        train_cls_loss /= batch_idx + 1
        train_reg_loss /= batch_idx + 1
        mean_train_acc = np.mean(train_acc)

        test_cls_loss = 0
        test_reg_loss = 0
        test_acc = []
        model.eval()
        for batch_idx, batch in enumerate(test_loader):
            batch = map(lambda d: d.to(args.device), batch)
            with torch.no_grad():
                loss, loss_info = model.compute_loss(*batch)

            test_cls_loss += loss_info['cls_loss']
            test_reg_loss += loss_info['reg_loss']
            test_acc.append(loss_info['acc'])
        model.train()

        test_cls_loss /= batch_idx + 1
        test_reg_loss /= batch_idx + 1
        mean_test_acc = np.concatenate(test_acc).mean()

        data.append(dict(epoch=epoch,
                         time_elapsed=time.perf_counter() - time_before_epoch,
                         train_cls_loss=train_cls_loss,
                         train_reg_loss=train_reg_loss,
                         test_cls_loss=test_cls_loss,
                         test_reg_loss=test_reg_loss,
                         mean_train_acc=mean_train_acc,
                         mean_test_acc=mean_test_acc,
                         lr=lr_scheduler.get_last_lr()[0],
                        ))
        with open(os.path.join(args.fdir, 'data.txt'), 'a') as f:
            f.write(str(data[-1])+'\n')

        log_str = f"Epoch {epoch+1}/{args.num_epochs} | " \
                  + f"CLS_LOSS={train_cls_loss:.4f}<{test_cls_loss:.4f}> " \
                  + f"REG_LOSS={train_reg_loss:.4f}<{test_reg_loss:.4f}> | " \
                  + f"ACC={mean_train_acc:.2f}<{mean_test_acc:.2f}> " \
                  + f"time={time.perf_counter() - time_before_epoch:.1f}s | " \
                  + f"lr={lr_scheduler.get_last_lr()[0]:.1e}"
        logger.info(log_str)
        time_before_epoch = time.perf_counter()
        lr_scheduler.step()

    if args.task == 'pose':
        eval_data = evaluate_pose_acc(model, test_loader, args.class_names)
    elif args.task == 'classification':
        eval_data = evaluate_classification(model, test_loader, args.class_names)

    np.save(os.path.join(args.fdir, "eval.npy"), np.array(eval_data))
    model.save(os.path.join(args.fdir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--ico_level', type=int, default=1)
    parser.add_argument('--sphere_fdim', type=int, default=128)
    parser.add_argument('--offset_lambda', type=float, default=100)
    parser.add_argument('--num_epochs', type=int, default=40)

    parser.add_argument('--lr_initial', type=float, default=0.001)
    parser.add_argument('--lr_step_size', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--sgd_nesterov', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--dataset_path', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='modelnet40',
                        choices=['modelnet40', 'shapenet55'])
    parser.add_argument('--task', type=str, default='pose',
                        choices=['pose', 'classification'])
    parser.add_argument('--objects', nargs='+', default=["select"],
                       help='specify list of class names or use "select" to use the same classes from paper')
    parser.add_argument('--img_mode', type=str, default='depth')

    parser.add_argument('--depth_shift_aug', type=float, default=0.05)
    parser.add_argument('--pixel_shift_aug', type=float, default=0.02)

    parser.add_argument('--encoder_type', type=str, default='resnet_equiv',
                        choices={'resnet', 'resnet_equiv'})
    parser.add_argument('--encoder_order', type=int, default=4)

    parser.add_argument('--num_workers', type=int, default=4,
                        help='workers used by dataloader')
    args = parser.parse_args()

    main(args)

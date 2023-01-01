from helper_tool import ConfigS3DIS as cfg
from RandLANet2 import Network, compute_loss, compute_plane_loss, compute_acc, IoUCalculator, visual_plane_error
from S3DIS_dataset2 import S3DIS, ActiveLearningSampler
import numpy as np
import os, argparse
from helper_ply import write_ply

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time
# from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
from sklearn.metrics import precision_recall_curve

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--load_checkpoint', default=False)
# parser.add_argument('--mode', default='train')
parser.add_argument('--load_checkpoint_path', default='output/new_10000_plane5_0.1_egc_up_nor/new_10000_plane5_0.1_egc_up_nor_149,tar', help='Model checkpoint path [default: None]')
parser.add_argument('--checkpoint_path', default='output/new_10000_plane5_0.1_egc_up_nor/new_10000_plane5_0.1_egc_up_nor_149,tar')
parser.add_argument('--log_dir', default='output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 8]')
parser.add_argument('--plot_output', default='output/test')
FLAGS = parser.parse_args()
#################################################   log   #################################################
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
# writer = SummaryWriter(FLAGS.plot_output)

# torch.manual_seed(52)
# np.random.seed(52)
# if args.cuda:
#     torch.cuda.manual_seed(52)

train_time = time.strftime('log_%Y-%m-%d_%H-%M-%S', time.gmtime())
train_file = 'train_' + train_time + '.txt'
test_file = 'test_' + train_time + '.txt'

def log_string(out_str, out_file):
    LOG_FOUT = open(os.path.join(LOG_DIR, out_file), 'a')
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

#################################################   dataset   #################################################
# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
DATASET = S3DIS(5)
# TEST_DATASET = S3DIS('validation')
TRAIN_DATASET = ActiveLearningSampler(DATASET, batch_size=FLAGS.batch_size, split='training')
TEST_DATASET = ActiveLearningSampler(DATASET, batch_size=cfg.val_batch_size, split='validation')

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=FLAGS.batch_size, collate_fn=TRAIN_DATASET.collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfg.val_batch_size, collate_fn=TEST_DATASET.collate_fn)

#################################################   network   #################################################

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Network(cfg, DATASET.input_names)
net.cuda()

# Load the Adam optimizer
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.learning_rate)
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=cfg.learning_rate, momentum=0.95)
# optimizer = optim.Adam([{'params': net.plane_refine.parameters(), 'lr':cfg.learning_rate_plane_refine}])

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if FLAGS.load_checkpoint:
    CHECKPOINT_PATH = FLAGS.load_checkpoint_path
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        # model_dict = {k: v for k,v in checkpoint['model_state_dict'].items() if 'plane_refine' not in k}
        # net.load_state_dict(model_dict, strict=False)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch), train_file)

# if torch.cuda.device_count() > 1:
#     print(f"Let's use {torch.cuda.device_count()}GPUs!")
#     dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     net = nn.DataParallel(net,[0,1])


#################################################   training functions   ###########################################


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * cfg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(optimizer):
    stat_dict = {}  # collect statistics
    stat_dict2 = {}  # collect statistics
    step = 0
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()  # set model to training mode val()
    iou_calc = IoUCalculator(cfg)
    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()
        step += 1
        # Forward pass
        optimizer.zero_grad()
        end_points = net(batch_data, split='training')

        loss, end_points = compute_loss(end_points, cfg)
        if cfg.plane_refine or cfg.plane_refine_4 or cfg.plane_refine_5 or cfg.plane_refine_6 or cfg.plane_refine_7:
            loss_plane, end_points = compute_plane_loss(end_points)
            if 'plane_loss' not in stat_dict2:
                stat_dict2['plane_loss'] = 0
            # FIXME possible not accurate
            stat_dict2['plane_loss'] += loss_plane.item()

            loss += loss_plane
        loss.backward()
        optimizer.step()

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key or 'plane_acc' in key or 'plane_loss' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                    stat_dict2[key] = 0
                stat_dict[key] += end_points[key].item()
                stat_dict2[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1), train_file)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval), train_file)
                stat_dict[key] = 0

    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100), train_file)
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string('-' * len(s), train_file)
    log_string(s, train_file)
    log_string('-' * len(s) + '\n', train_file)

    # writer.add_scalars("Train/miou", {'mean_iou': mean_iou * 100}, epoch)
    # writer.add_scalars("Train/acc", { 'acc': stat_dict2['acc'] / step}, epoch)
    # writer.add_scalars("Train/loss", {'loss': stat_dict2['loss'] / step}, epoch)

    # if cfg.plane_refine or cfg.plane_refine_4 or cfg.plane_refine_5 or cfg.plane_refine_6 or cfg.plane_refine_7:
    #     writer.add_scalars("Train/loss", {'plane_loss': stat_dict2['plane_loss'] / step}, epoch)
    #     writer.add_scalars("Train/acc", {'plane_acc': stat_dict2['plane_acc'] / step}, epoch)
    #
    # writer.close()

def evaluate_one_epoch(epoch):
    stat_dict = {} # collect statistics
    net.eval() # set model to eval mode (for bn and dp)
    iou_calc = IoUCalculator(cfg)
    stat_dict2 = {}  # collect statistics
    step = 0

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        # Forward pass
        step += 1
        with torch.no_grad():
            end_points = net(batch_data, split='validation')

        loss, end_points = compute_loss(end_points, cfg)
        if cfg.plane_refine or cfg.plane_refine_4 or cfg.plane_refine_5 or cfg.plane_refine_6  or cfg.plane_refine_7:
            loss_plane, end_points = compute_plane_loss(end_points)
            if 'plane_loss' not in stat_dict2:
                stat_dict2['plane_loss'] = 0
            # FIXME possible not accurate
            stat_dict2['plane_loss'] += loss_plane.item()
            loss += loss_plane

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key or 'plane_loss' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                    stat_dict2[key] = 0
                stat_dict[key] += end_points[key].item()
                stat_dict2[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1), train_file)

    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))), train_file)
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100), train_file)
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string('-' * len(s), train_file)
    log_string(s, train_file)
    log_string('-' * len(s) + '\n', train_file)

    # writer.add_scalars("Eval/miou", {'mean_iou': mean_iou * 100}, epoch)
    # writer.add_scalars("Eval/acc", {'acc': stat_dict2['acc'] / step}, epoch)
    # writer.add_scalars("Eval/loss", {'loss': stat_dict2['loss'] / step}, epoch)
    # if cfg.plane_refine or cfg.plane_refine_4 or cfg.plane_refine_5 or cfg.plane_refine_6 or cfg.plane_refine_7:
    #     writer.add_scalars("Eval/loss", {'plane_loss': stat_dict2['plane_loss'] / step}, epoch)
    #     writer.add_scalars("Eval/acc", {'plane_acc': stat_dict2['plane_acc'] / step}, epoch)
    # writer.close()

def train(start_epoch):
    global EPOCH_CNT
    global OPTMIZER
    OPTMIZER = optimizer
    loss = 0
    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        # if epoch == 150:
        #     OPTMIZER = optimizer2
        log_string('**** EPOCH %03d ****' % (epoch), train_file)

        log_string(str(datetime.now()), train_file)

        # np.random.seed()
        train_one_epoch(OPTMIZER)

        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            log_string('**** EVAL EPOCH %03d START****' % (epoch), train_file)
            evaluate_one_epoch(epoch)
            log_string('**** EVAL EPOCH %03d END****' % (epoch), train_file)
            # Save checkpoint
            save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }
            try: # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(FLAGS.log_dir, FLAGS.log_dir.split('/')[1] +'_'+ str(epoch) + '.tar'))

def test(num_votes=100):
    # Smoothing parameter for votes
    test_smooth = 0.95

    saving_path = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    name = FLAGS.load_checkpoint_path.split('/')[1]
    test_path = os.path.join('test', name, saving_path)
    os.makedirs(test_path) if not os.path.exists(test_path) else None
    os.makedirs(os.path.join(test_path, 'val_preds')) if not os.path.exists(os.path.join(test_path, 'val_preds')) else None

    step_id = 0
    epoch_id = 0
    last_min = -0.5

    net.eval()
    test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32)
                       for l in TEST_DATASET.dataset.input_labels['validation']]
    test_plane_probs = [np.zeros(shape=[l.shape[0], 1], dtype=np.float32)
                       for l in TEST_DATASET.dataset.input_labels['validation']]
    test_plane_preds = [np.zeros(shape=[l.shape[0], 1], dtype=np.float32)
                        for l in TEST_DATASET.dataset.input_labels['validation']]

    val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
    i = 0
    for label_val in TEST_DATASET.dataset.label_values:
        if label_val not in TEST_DATASET.dataset.ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in TEST_DATASET.dataset.val_labels])
            i += 1

    while last_min < num_votes:

        for batch_idx, batch_data in enumerate(TEST_DATALOADER):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            # Forward pass
            with torch.no_grad():
                end_points = net(batch_data, split='validation')
            _, end_points = compute_loss(end_points, cfg)

            acc, end_points = compute_acc(end_points)
            loss, end_points = compute_plane_loss(end_points)
            end_points = visual_plane_error(end_points)
            #
            print('step' + str(step_id) + ' acc:' + str(acc) + '  plane_acc'+ str(end_points['plane_acc']))
            # print('step' + str(step_id) + ' acc:' + str(acc))

            # TODO find result stacked_probs
            stacked_probs = end_points['valid_logits'].cpu().numpy()
            stacked_plane = end_points['plane_visual'].cpu().numpy()
            stacked_plane_pred = end_points['plane_visual_pred'].cpu().numpy()
            point_idx = end_points['input_inds'].cpu().numpy()
            cloud_idx = end_points['cloud_inds'].cpu().numpy()

            stacked_probs = np.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                                   cfg.num_classes])
            stacked_plane = np.reshape(stacked_plane, [cfg.val_batch_size, cfg.num_points, 1])
            stacked_plane_pred = np.reshape(stacked_plane_pred, [cfg.val_batch_size, cfg.num_points, 1])

            for j in range(np.shape(stacked_probs)[0]):
                probs = stacked_probs[j, :, :]
                planes = stacked_plane[j, :, :]
                planes_pred = stacked_plane_pred[j, :, :]

                p_idx = point_idx[j, :]
                c_i = cloud_idx[j][0]

                test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                test_plane_probs[c_i][p_idx] = planes
                test_plane_preds[c_i][p_idx] = planes_pred
            step_id += 1

        new_min = np.min(TEST_DATASET.min_possibility['validation'])
        log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), test_file)

        if last_min + 1 < new_min:

            # Update last_min
            last_min += 1

            # Show vote results (On subcloud so it is not the good values here)
            log_string('\nConfusion on sub clouds', test_file)
            confusion_list = []

            num_val = len(TEST_DATASET.dataset.input_labels['validation'])
            # every room
            for i_test in range(num_val):
                probs = test_probs[i_test]
                preds = TEST_DATASET.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                labels = TEST_DATASET.dataset.input_labels['validation'][i_test]

                # Confs
                confusion_list += [confusion_matrix(labels, preds, TEST_DATASET.dataset.label_values)]

            # Regroup confusions
            C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

            # Rescale with the right number of point per class
            C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

            # Compute IoUs
            IoUs = DP.IoU_from_confusions(C)
            m_IoU = np.mean(IoUs)
            s = '{:5.2f} | '.format(100 * m_IoU)
            for IoU in IoUs:
                s += '{:5.2f} '.format(100 * IoU)
            log_string(s + '\n', test_file)

            if int(np.ceil(new_min)) % 1 == 0:

                # Project predictions
                log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), test_file)
                proj_probs_list = []
                proj_plane_probs_list = []
                proj_plane_preds_list = []
                for i_val in range(num_val):
                    # Reproject probs back to the evaluations points
                    proj_idx = TEST_DATASET.dataset.val_proj[i_val]
                    probs = test_probs[i_val][proj_idx, :]
                    planes = test_plane_probs[i_val][proj_idx, :]
                    plane_preds = test_plane_preds[i_val][proj_idx, :]
                    proj_plane_probs_list += [planes]
                    proj_plane_preds_list += [plane_preds]
                    proj_probs_list += [probs]

                # Show vote results
                log_string('Confusion on full clouds', test_file)
                confusion_list = []
                confusion_boundary_list = []
                for i_test in range(num_val):
                    # Get the predicted labels
                    preds = TEST_DATASET.dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                    probs = np.max(proj_probs_list[i_test], axis=1)

                    # Confusion
                    labels = TEST_DATASET.dataset.val_labels[i_test]
                    plane_results = proj_plane_probs_list[i_test]
                    plane_preds = proj_plane_preds_list[i_test]
                    mask = TEST_DATASET.dataset.val_boundarys[i_test] == 1

                    acc = np.sum(preds == labels) / len(labels)
                    boundary_acc = np.sum(preds[mask] == labels[mask]) / len(labels[mask])
                    log_string(TEST_DATASET.dataset.input_names['validation'][i_test] + ' Acc:' + str(acc) + '  Boundary Acc:' + str(boundary_acc),
                               test_file)

                    confusion_list += [confusion_matrix(labels, preds, TEST_DATASET.dataset.label_values)]
                    confusion_boundary_list += [confusion_matrix(labels[mask], preds[mask], TEST_DATASET.dataset.label_values)]
                    name = TEST_DATASET.dataset.input_names['validation'][i_test] + '.ply'
                    write_ply(os.path.join(test_path, name), [preds, labels, probs, plane_results, plane_preds], ['pred', 'label',
                                                                                           'probs', 'plane_result', 'plane_pred'])
                    # write_ply(os.path.join(test_path, name), [preds, labels, probs], ['pred', 'label',
                    #                                                                        'probs'])

                # Regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0)
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string('-' * len(s), test_file)
                log_string(s, test_file)
                log_string('-' * len(s) + '\n', test_file)

                C = np.sum(np.stack(confusion_boundary_list), axis=0)
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string('-' * len(s), test_file)
                log_string(s, test_file)
                log_string('-' * len(s) + '\n', test_file)

                F1_Scores = DP.F1_from_confusions(C)
                m_F1 = np.mean(F1_Scores)
                s = '{:5.2f} | '.format(100 * m_F1)
                for F1 in F1_Scores:
                    s += '{:5.2f} '.format(100 * F1)

                log_string('-' * len(s), test_file)
                log_string(s, test_file)
                log_string('-' * len(s) + '\n', test_file)

                print('finished \n')

                return
            epoch_id += 1
            step_id = 0
            continue
    return

def test_bound(num_votes=100):
    # Smoothing parameter for votes
    test_smooth = 0.95

    saving_path = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    name = FLAGS.checkpoint_path.split('/')[1].split('.')[0]
    test_path = os.path.join('test', name, saving_path)
    os.makedirs(test_path) if not os.path.exists(test_path) else None
    os.makedirs(os.path.join(test_path, 'val_preds')) if not os.path.exists(
        os.path.join(test_path, 'val_preds')) else None

    step_id = 0
    epoch_id = 0
    last_min = -0.5

    net.eval()
    test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32)
                  for l in TEST_DATASET.dataset.input_labels['validation']]

    val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
    i = 0
    for label_val in TEST_DATASET.dataset.label_values:
        if label_val not in TEST_DATASET.dataset.ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in TEST_DATASET.dataset.val_labels])
            i += 1

    while last_min < num_votes:

        for batch_idx, batch_data in enumerate(TEST_DATALOADER):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            # Forward pass
            with torch.no_grad():
                end_points = net(batch_data, split='validation')
            _, end_points = compute_loss(end_points, cfg)

            acc, end_points = compute_acc(end_points)

            print('step' + str(step_id) + ' acc:' + str(acc))

            # TODO find result stacked_probs
            stacked_probs = torch.nn.functional.sigmoid(end_points['valid_logits']).cpu().numpy()
            point_idx = end_points['input_inds'].cpu().numpy()
            cloud_idx = end_points['cloud_inds'].cpu().numpy()
            stacked_probs = np.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                                       cfg.num_classes])

            for j in range(np.shape(stacked_probs)[0]):
                probs = stacked_probs[j, :, :]
                p_idx = point_idx[j, :]
                c_i = cloud_idx[j][0]
                test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs
            step_id += 1

        new_min = np.min(TEST_DATASET.min_possibility['validation'])
        log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), test_file)

        if last_min + 1 < new_min:

            # Update last_min
            last_min += 1


            num_val = len(TEST_DATASET.dataset.input_labels['validation'])


            if int(np.ceil(new_min)) % 1 == 0:

                # Project predictions
                log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), test_file)
                proj_probs_list = []

                for i_val in range(num_val):
                    # Reproject probs back to the evaluations points
                    proj_idx = TEST_DATASET.dataset.val_proj[i_val]
                    probs = test_probs[i_val][proj_idx, :]
                    proj_probs_list += [probs]


                print('MF on full clouds')
                t1 = time.time()
                class_ind = 0  # 0 ~ num_class - 1: indicating channel number of outputs
                class_MFs = []

                for l_ind, label_value in enumerate(TEST_DATASET.dataset.label_values):
                    if label_value not in TEST_DATASET.dataset.ignored_labels:

                        proj_probs_class = []

                        for proj_p in proj_probs_list:
                            proj_probs_class += [proj_p[:, class_ind]]

                        preds_class = []
                        target_class = []

                        for i_test in range(num_val):
                            # prediction for class_i
                            pre_class = proj_probs_class[i_test]
                            # boundaries_class: indicating if this point is boundary for class i
                            boundaries_class_0 = [(l == label_value) for l in
                                                  TEST_DATASET.dataset.validation_b_c_0[i_test]]
                            boundaries_class_1 = [(l == label_value) for l in
                                                  TEST_DATASET.dataset.validation_b_c_1[i_test]]
                            boundaries_class_2 = [(l == label_value) for l in
                                                  TEST_DATASET.dataset.validation_b_c_2[i_test]]
                            boundaries_class = [(l0 or l1 or l2) for l0, l1, l2 in
                                                zip(boundaries_class_0, boundaries_class_1, boundaries_class_2)]

                            if len(TEST_DATASET.dataset.ignored_labels) > 0:
                                # Boolean mask of points that should be ignored
                                ignored_bool = np.zeros_like(TEST_DATASET.dataset.validation_labels[i_test],
                                                             dtype=np.bool)
                                for ign_val in TEST_DATASET.dataset.ignored_labels:
                                    ignored_bool = np.logical_or(ignored_bool,
                                                                 np.equal(
                                                                     TEST_DATASET.dataset.validation_labels[i_test],
                                                                     ign_val))

                                # inds that are not ignored
                                inds = np.squeeze(np.where(np.logical_not(ignored_bool)))

                                # select points that are not ignored
                                pre_class = np.array(pre_class)[inds]
                                boundaries_class = np.array(boundaries_class)[inds]

                            preds_class = np.hstack((preds_class, pre_class))
                            target_class = np.hstack((target_class, boundaries_class))

                        if np.sum(target_class) == 0:
                            raise ValueError('This class does not exist in the testing set')
                        precisions, recalls, thresholds = precision_recall_curve(target_class, preds_class)
                        f1_scores = []
                        for (precision, recall) in zip(precisions, recalls):
                            if recall + precision == 0:
                                f1_scores += [0.0]
                            else:
                                f1_scores += [2 * recall * precision / (recall + precision)]
                        # class_MF: maximal F meature for class i
                        class_MF = np.max(f1_scores)
                        print('class_{}:{}'.format(class_ind, class_MF))
                        class_MFs += [class_MF]
                        class_ind = class_ind + 1

                t2 = time.time()
                print('Done in {:.1f} s\n'.format(t2 - t1))

                mMF = np.mean(class_MFs)
                s = '{:5.2f} | '.format(100 * mMF)
                for MF in class_MFs:
                    s += '{:5.2f} '.format(100 * MF)
                print('-' * len(s))
                print(s)
                print('-' * len(s) + '\n')
                return
            epoch_id += 1
            step_id = 0
            continue
    return

def test2(num_votes=100):
    # Smoothing parameter for votes
    test_smooth = 0.95

    saving_path = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    name = FLAGS.load_checkpoint_path.split('/')[1].split('.')[0]
    test_path = os.path.join('test', name, saving_path)
    os.makedirs(test_path) if not os.path.exists(test_path) else None
    os.makedirs(os.path.join(test_path, 'val_preds')) if not os.path.exists(os.path.join(test_path, 'val_preds')) else None

    step_id = 0
    epoch_id = 0
    last_min = -0.5

    net.eval()
    test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32)
                       for l in TEST_DATASET.dataset.input_labels['validation']]


    val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
    i = 0
    for label_val in TEST_DATASET.dataset.label_values:
        if label_val not in TEST_DATASET.dataset.ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in TEST_DATASET.dataset.val_labels])
            i += 1

    while last_min < num_votes:

        for batch_idx, batch_data in enumerate(TEST_DATALOADER):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            # Forward pass
            with torch.no_grad():
                end_points = net(batch_data, split='validation')
            _, end_points = compute_loss(end_points, cfg)

            acc, end_points = compute_acc(end_points)


            print('step' + str(step_id) + ' acc:' + str(acc) )

            # TODO find result stacked_probs
            stacked_probs = end_points['valid_logits'].cpu().numpy()

            point_idx = end_points['input_inds'].cpu().numpy()
            cloud_idx = end_points['cloud_inds'].cpu().numpy()

            stacked_probs = np.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                                   cfg.num_classes])


            for j in range(np.shape(stacked_probs)[0]):
                probs = stacked_probs[j, :, :]

                p_idx = point_idx[j, :]
                c_i = cloud_idx[j][0]

                test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs

            step_id += 1

        new_min = np.min(TEST_DATASET.min_possibility['validation'])
        log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), test_file)

        if last_min + 1 < new_min:

            # Update last_min
            last_min += 1

            # Show vote results (On subcloud so it is not the good values here)
            log_string('\nConfusion on sub clouds', test_file)
            confusion_list = []

            num_val = len(TEST_DATASET.dataset.input_labels['validation'])
            # every room
            for i_test in range(num_val):
                probs = test_probs[i_test]
                preds = TEST_DATASET.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                labels = TEST_DATASET.dataset.input_labels['validation'][i_test]

                # Confs
                confusion_list += [confusion_matrix(labels, preds, TEST_DATASET.dataset.label_values)]

            # Regroup confusions
            C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

            # Rescale with the right number of point per class
            C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

            # Compute IoUs
            IoUs = DP.IoU_from_confusions(C)
            m_IoU = np.mean(IoUs)
            s = '{:5.2f} | '.format(100 * m_IoU)
            for IoU in IoUs:
                s += '{:5.2f} '.format(100 * IoU)
            log_string(s + '\n', test_file)

            if int(np.ceil(new_min)) % 1 == 0:

                # Project predictions
                log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), test_file)
                proj_probs_list = []
                proj_plane_probs_list = []
                proj_plane_preds_list = []
                for i_val in range(num_val):
                    # Reproject probs back to the evaluations points
                    proj_idx = TEST_DATASET.dataset.val_proj[i_val]
                    probs = test_probs[i_val][proj_idx, :]


                    proj_probs_list += [probs]

                # Show vote results
                log_string('Confusion on full clouds', test_file)
                confusion_list = []
                confusion_boundary_list = []
                for i_test in range(num_val):
                    # Get the predicted labels
                    preds = TEST_DATASET.dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                    probs = np.max(proj_probs_list[i_test], axis=1)

                    # Confusion
                    labels = TEST_DATASET.dataset.val_labels[i_test]

                    mask = TEST_DATASET.dataset.val_boundarys[i_test] == 1

                    acc = np.sum(preds == labels) / len(labels)
                    boundary_acc = np.sum(preds[mask] == labels[mask]) / len(labels[mask])
                    log_string(TEST_DATASET.dataset.input_names['validation'][i_test] + ' Acc:' + str(acc) + '  Boundary Acc:' + str(boundary_acc),
                               test_file)

                    confusion_list += [confusion_matrix(labels, preds, TEST_DATASET.dataset.label_values)]
                    confusion_boundary_list += [confusion_matrix(labels[mask], preds[mask], TEST_DATASET.dataset.label_values)]
                    name = TEST_DATASET.dataset.input_names['validation'][i_test] + '.ply'
                    write_ply(os.path.join(test_path, name), [preds, labels, probs], ['pred', 'label',
                                                                                           'probs'])

                # Regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0)
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string('-' * len(s), test_file)
                log_string(s, test_file)
                log_string('-' * len(s) + '\n', test_file)

                C = np.sum(np.stack(confusion_boundary_list), axis=0)
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string('-' * len(s), test_file)
                log_string(s, test_file)
                log_string('-' * len(s) + '\n', test_file)

                F1_Scores = DP.F1_from_confusions(C)
                m_F1 = np.mean(F1_Scores)
                s = '{:5.2f} | '.format(100 * m_F1)
                for F1 in F1_Scores:
                    s += '{:5.2f} '.format(100 * F1)

                log_string('-' * len(s), test_file)
                log_string(s, test_file)
                log_string('-' * len(s) + '\n', test_file)

                print('finished \n')

                return
            epoch_id += 1
            step_id = 0
            continue
    return

if __name__ == '__main__':
    train(start_epoch)
    # test2()


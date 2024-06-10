import argparse
import os
import datetime
#import ref

class Opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.init()
    self.opt = self.parser.parse_args()

    self.opt.output_dir = os.path.join(self.opt.exp_dir, self.opt.exp_id)
    if not os.path.exists(self.opt.output_dir):
      os.makedirs(self.opt.output_dir)

  def init(self):
    #{{{
    self.parser.add_argument('--fl', dest='no_fl', action='store_false', help='use fl')
    self.parser.add_argument('--no-fl', dest='no_fl', action='store_true', help='no fl')
    self.parser.set_defaults(no_fl=True)

    self.parser.add_argument('--dist-num', '--dist_num', type=int, default=3, help="number of parties")
    self.parser.add_argument('--dist-rank', '--dist_rank', type=int, default=0, help='rank of parties')
    self.parser.add_argument('--master-addr', '--master_addr', type=str, default='172.20.117.210', help='master address')
    self.parser.add_argument('--master-port', '--master_port', type=int, default=29500, help='master port')

    self.parser.add_argument('--exp-id', '--exp_id', default = 'default', help = 'Experiment ID')
    self.parser.add_argument('--exp-dir', '--exp_dir', default = 'exp', help = 'Experiment ID')
    self.parser.add_argument('-test', action = 'store_true', help = 'test')
    #self.parser.add_argument('-DEBUG', type = int, default = 0, help = 'DEBUG level')
    self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')

    self.parser.add_argument('-resume', default=False, type=bool, metavar='BOOL', help='Use the checkpoint or not')
    self.parser.add_argument('-loadModel', default = None, help = 'Provide full path to a previously trained model')

    ## Train parameters
    self.parser.add_argument('-lr', type = float, default = 3e-1, help = 'Learning Rate')
    self.parser.add_argument('-epochs', type = int, default = 250, help = '#training epochs')
    self.parser.add_argument('-trainBatch', type = int, default = 8, help = 'Mini-batch size')
    self.parser.add_argument('--batch-size', '--batch_size', type = int, default = 32, help = 'batch size')

    self.parser.add_argument('--clear-cache', '--clear_cache', default=False, type=bool, metavar='BOOL', help='Clear dataset cache')

    ## Visdom
    self.parser.add_argument('--plot-server',  '--plot_server', type=str, default='http://10.10.10.100', help='IP address')
    self.parser.add_argument('--exp-name', '--exp_name', type=str, default='lstm_gaze', help='The env name in visdom')
    self.parser.add_argument('--plot-port', '--plot_port', type=int, default=31831, help='Port number')
    self.parser.add_argument('--save-interval', '--save_interval', type=int, default=1, help='Port number')


    self.parser.add_argument('--snapshot-fname-prefix', '--snapshot_fname_prefix', default='', type=str, metavar='PATH', help='path to snapshot')

    self.parser.add_argument('--sal-image-fname-dir', '--sal_image_fname_dir', default='exps/', type=str, metavar='PATH', help='path to sal image')
    self.parser.add_argument('--epoch-st', '--epoch_st', default=0, type=int, help='rank of distributed processes')
    self.parser.add_argument('--epoch-end', '--epoch_end', default=250, type=int, help='rank of distributed processes')

    self.parser.add_argument('--debug', dest='debug', help='Set to True for forward network.', action='store_true', default=False)
    self.parser.add_argument('--eval', dest='eval', help='Set to True for forward network.', action='store_true', default=False)

    self.parser.add_argument('--use-vis', '--use_vis', dest='use_vis',help='use vis', action='store_true',default=False)

    # Mode
    self.parser.add_argument('--mode', type=str, default='train', help='The mode name: pretrain_tnet, pretrain_mnet, end_to_end, test')
    self.parser.add_argument('--patch-size', '--patch_size', type=int, default=256, help='patch size for train')
    self.parser.add_argument('--data-dir', '--data_dir', type=str, default='/media/astar/e006bf52-80e3-47f3-b5ca-f5871c5a5e7f/home/astar/FL_Platform/OCT/CellData/OCT/', help='dataset directory')
    self.parser.add_argument('--data-sub-dir', '--data_sub_dir', type=str, default=None, help='dataset sub dir')

    self.parser.add_argument('--model-type', '--model_type', type=str, default='ViT', help='dataset directory')   # resnet50

    self.parser.add_argument('--use-grad-cam', '--use_grad_cam', dest='use_grad_cam',help='use grad cam', action='store_true',default=False)
    self.parser.add_argument('--use-tensorboard', '--use_tensorboard', dest='use_tensorboard',help='use tensorboard', action='store_true',default=False)
    self.parser.add_argument('--use-grad-cam-layers', '--use_grad_cam_layers', dest='use_grad_cam_layers',help='use grad cam', action='store_true',default=False)

    self.parser.add_argument('--epsilon', type=float, default=0, help='patch size for train')
    self.parser.add_argument('--adv-dataset-mode', '--adv_dataset_mode', type=str, default='train', help='The mode name: pretrain_tnet, pretrain_mnet, end_to_end, test')

    self.parser.add_argument('--requires-control', '--requires_control', dest='requires_control',help='use grad cam', action='store_true',default=False)
    self.parser.add_argument('--is-defense', '--is_defense', dest='is_defense',help='use grad cam', action='store_true', default=False)

    self.parser.add_argument('--use-clean-eval', '--use_clean_eval', dest='use_clean_eval',help='use grad cam', action='store_true',default=False)
    self.parser.add_argument('--use-multi-epsilon', '--use_multi_epsilon', dest='use_multi_epsilon',help='use grad cam', action='store_true', default=False)

    self.parser.add_argument('--dataset-type', '--dataset_type', type=str, default='x-ray', help='dataset directory')

    self.parser.add_argument('--num-of-tasks', '--num_of_tasks', type=int, default=14, help='patch size for train')
    self.parser.add_argument('--use-whole-dataset', '--use_whole_dataset', dest='use_whole_dataset',help='use grad cam', action='store_true',default=False)
    self.parser.add_argument('--noise-multiplier', '--noise_multiplier', type=float, default=0.5, help='dp noise multiplier')

    #self.parser.add_argument('--use_multi_epsilon', dest='use_multi_epsilon',help='use grad cam', action='store_true',default=False)
    #}}}

  def log(self):
    args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                if not name.startswith('_'))
    #refs = dict((name, getattr(ref, name)) for name in dir(ref)
    #            if not name.startswith('_'))

    logger.print('\nArgs:')
    for k, v in sorted(args.items()):
      logger.print('%s,%s' % (str(k), str(v)))

    #opt_file.write('==> Args:\n')
    #for k, v in sorted(refs.items()):
    #   opt_file.write('  %s: %s\n' % (str(k), str(v)))

opts = Opts()
opt = opts.opt

from . utils.logger import logger
opts.log()

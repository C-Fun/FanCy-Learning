import os
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


def compute_mac(model, im_size, log_file=None):
	h_in, w_in = im_size

	macs = []
	for name, l in model.named_modules():
		if isinstance(l, nn.Conv2d):
			c_in    = l.in_channels
			k       = l.kernel_size[0]
			h_out   = int((h_in-k+2*l.padding[0])/(l.stride[0])) + 1
			w_out   = int((w_in-k+2*l.padding[0])/(l.stride[0])) + 1
			c_out   = l.out_channels
			mac     = k*k*c_in*h_out*w_out*c_out
			if mac == 0:
				pdb.set_trace()
			macs.append(mac)
			h_in    = h_out
			w_in    = w_out
			print('{}, Mac:{}'.format(name, mac))
		if isinstance(l, nn.Linear):
			mac     = l.in_features * l.out_features
			macs.append(mac)
			print('{}, Mac:{}'.format(name, mac))
		if isinstance(l, nn.AvgPool2d):
			h_in    = h_in//l.kernel_size
			w_in    = w_in//l.kernel_size
	print('Mac: {:e}'.format(sum(macs)))
	exit()
	if log_file is not None:
		log_file.write('\n\n Mac: {}'.format(sum(macs)))


def load_pretrained_weight(model, pretrained_file, log_file=None):
	'''
	state=torch.load(args.pretrained_ann, map_location='cpu')
	cur_dict = model.state_dict()
	for key in state['state_dict'].keys():
		if key in cur_dict:
			if (state['state_dict'][key].shape == cur_dict[key].shape):
				cur_dict[key] = nn.Parameter(state[key].data)
				f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
			else:
				f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
		else:
			f.write('\n Error: Loaded weight {} not present in current model'.format(key))

	#model.load_state_dict(cur_dict)
	'''
	state = torch.load(pretrained_file, map_location='cpu')

	missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
	if log_file is not None:
		log_file.write('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))
		log_file.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))
	return model


class RunManager(object):
	def __init__(self, model, optimizer, learning_rate, lr_interval, lr_reduce, log_file=None):
		self.model = model
		self.learning_rate = learning_rate
		if optimizer == 'SGD':
			self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, momentum=0.9, weight_decay=0.000)
		elif optimizer == 'Adam':
			self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, amsgrad=True, weight_decay=0.000)

		self.lr_interval = lr_interval
		self.lr_reduce = lr_reduce
		self.log_file = log_file
		if self.log_file is not None:
			self.log_file.write('\n\n Optimizer: {}'.format(self.optimizer))
		self.max_accuracy = 0
		self.start_time = None
		self.save_path = None

	def set_start_time(self, time):
		self.start_time = time

	def train(self, epoch, loader):
		losses = AverageMeter('Loss')
		top1   = AverageMeter('Acc@1')

		if epoch in self.lr_interval:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = param_group['lr'] / self.lr_reduce
				self.learning_rate = param_group['lr']

		# total_correct = 0
		self.model.train()
		with tqdm(loader, total=len(loader)) as t:
			for batch_idx, (data, target) in enumerate(t):
				if torch.cuda.is_available():
					data, target = data.cuda(), target.cuda()

				self.optimizer.zero_grad()

				output = self.model(data) # regular model

				loss = F.cross_entropy(output, target)
				# loss.backward(inputs = list(self.model.parameters()))
				loss.backward()
				self.optimizer.step()

				pred = output.max(1,keepdim=True)[1]
				correct = pred.eq(target.data.view_as(pred)).cpu().sum()
				# total_correct += correct.item()

				losses.update(loss.item(), data.size(0))
				top1.update(correct.item()/data.size(0), data.size(0))

				if batch_idx % 1 == 0:
					# t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(loss.item(), correct.item()/data.size(0)))
					t.set_postfix_str("train_loss: {:.4f}, train_acc: {:.4f}".format(losses.avg, top1.avg))

		self.log_file.write('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
				epoch,
				self.learning_rate,
				losses.avg,
				top1.avg
				)
			)

	def test(self, epoch, loader, save=False, ann_path=None, identifier=None):
		losses = AverageMeter('Loss')
		top1   = AverageMeter('Acc@1')

		with torch.no_grad():
			self.model.eval()
			total_loss = 0
			correct = 0
			with tqdm(loader, total=len(loader)) as t:
				for batch_idx, (data, target) in enumerate(t):
					if torch.cuda.is_available():
						data, target = data.cuda(), target.cuda()

					output = self.model(data)

					loss = F.cross_entropy(output,target)
					total_loss += loss.item()

					pred = output.max(1, keepdim=True)[1]
					correct = pred.eq(target.data.view_as(pred)).cpu().sum()

					losses.update(loss.item(), data.size(0))
					top1.update(correct.item()/data.size(0), data.size(0))

					if batch_idx % 1 == 0:
						t.set_postfix_str("test_loss: {:.4f}, test_acc: {:.4f}".format(losses.avg, top1.avg))

		if top1.avg>self.max_accuracy:
			self.max_accuracy = top1.avg
			state = {
					'accuracy'      : self.max_accuracy,
					'epoch'         : epoch,
					'state_dict'    : self.model.state_dict(),
					'optimizer'     : self.optimizer.state_dict()
			}

			if save==True and ann_path is not None and identifier is not None:
				try:
					os.makedirs(ann_path)
				except OSError:
					pass
				filename = os.path.join(ann_path, f'{identifier}.pth')
				torch.save(state, filename)
				self.save_path = filename

		self.log_file.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.format(
			losses.avg,
			top1.avg,
			self.max_accuracy,
			datetime.timedelta(seconds=(datetime.datetime.now() - self.start_time).seconds)
			)
		)
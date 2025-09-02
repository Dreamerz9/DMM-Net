import os
import csv
import Loss
from models.Diffusionnet import *

from torch import optim
from evaluation import *
import torch.nn.functional as F
import torchvision

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.scheduler = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch

		# Losses
		self.criterion = Loss.DiceLoss()

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_test = config.num_epochs_test
		self.batch_size = config.batch_size

		# Path
		self.model_path = config.model_path
		self.train_result_path = config.train_result_path
		self.val_result_path = config.val_result_path
		self.test_result_path = config.test_result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=1,output_ch=1)
		elif self.model_type == 'MSU_Net':
			self.unet = MSU_Net(img_ch=1, output_ch=1)
		elif self.model_type == 'my_Net':
			self.unet =	Difnet()   #MSAM

		self.optimizer = optim.Adam(self.unet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)


		self.unet.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def reset_grad(self):
		# Zero the gradient buffers.
		self.unet.zero_grad()

	def train(self):
		"""Train encoder, generator and discriminator."""

		# ====================================== Training ===========================================#

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr))

		if os.path.isfile(unet_path):
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
		else:
			learn_rate = self.lr
			best_Dice = 0.

			print("Train......\n")
			for epoch in range(self.num_epochs):

				self.unet.train()
				epoch_loss = 0

				t_TPR = 0.  # TPR
				t_FPR = 0.  # FPR
				t_PPV = 0.  # PPV
				t_JS = 0.   # Jaccard Similarity
				t_DC = 0.   # Dice Coefficient
				t_SE = 0.   # SE
				t_SP = 0.   # SP
				# t_HD = 0.
				# t_Recall = 0.

				for i, (images, pet, label) in enumerate(self.train_loader):

					images = images.to(self.device)
					pet = pet.to(self.device)
					GT = label.to(self.device)
					SR, loss_re, loss_score_pet, loss_score_ct, psnr, ssim = self.unet(images, pet, num_modal=1)
					SR_flat = SR.view(SR.size(0), -1)

					GT_flat = GT.view(GT.size(0), -1)
					loss = self.criterion(SR_flat, GT_flat)
					total_loss = loss + 0.8 * (loss_re + loss_score_pet + loss_score_ct)
					epoch_loss += total_loss.item()

					# Backprop + optimize
					self.reset_grad()
					total_loss.backward()
					self.optimizer.step()

					t_TPR += get_TPR(SR, GT)
					t_FPR += get_FPR(SR, GT)
					t_PPV += get_precision(SR, GT)
					t_JS += get_JS(SR, GT)
					t_DC += get_DC(SR, GT)
					t_SE += get_sensitivity(SR, GT)
					t_SP += get_specificity(SR, GT)



				length = len(self.train_loader)
				t_TPR = t_TPR / length
				t_FPR = t_FPR / length
				t_PPV = t_PPV / length
				t_JS = t_JS / length
				t_DC = t_DC / length
				t_SE = t_SE / length
				t_SP = t_SP / length
				# t_HD = t_HD / length
				# t_Recall = t_Recall / length

				"""
				torchvision.utils.save_image(images.data.cpu(),
                                             os.path.join(self.train_result_path,
                                                          '%s_train_%d_image.jpg' % (self.model_type, epoch + 1)))
				torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.train_result_path,
                                                          '%s_train_%d_SR.jpg' % (self.model_type, epoch + 1)))
				torchvision.utils.save_image(GT.data.cpu(),
                                             os.path.join(self.train_result_path,
                                                          '%s_train_%d_GT.jpg' % (self.model_type, epoch + 1)))
                		"""
				
				print('Epoch [%d/%d], \n[Training]   TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE:%.4f, SP:%.4f'    # , HD:%.4f, Recall:%.4f
				      % (epoch + 1, self.num_epochs, t_TPR, t_FPR, t_PPV, t_JS, t_DC, t_SE, t_SP))   # , t_HD, t_Recall
				
				e = open(os.path.join(self.train_result_path, 'train_result.csv'), 'a', encoding='utf-8',newline='')
				wr = csv.writer(e)
				wr.writerow([self.model_type, t_TPR, t_FPR, t_PPV, t_JS, t_DC, t_SE, t_SP, epoch_loss, epoch + 1, self.num_epochs, learn_rate])   # , t_HD, t_Recall
				e.close()

				me = open(os.path.join(self.train_result_path, 'sim_result.csv'), 'a', encoding='utf-8',newline='')
				wf = csv.writer(me)
				wf.writerow([epoch + 1, psnr, ssim])
				me.close()
	#===================================== Validation ====================================#
				print("Valid......\n")
				self.unet.train(False)
				self.unet.eval()

				with torch.no_grad():
					v_TPR = 0.  # TPR
					v_FPR = 0.  # FPR
					v_PPV = 0.  # PPV
					v_JS = 0.  # Jaccard Similarity
					v_DC = 0.  # Dice Coefficient
					v_SE = 0.  # SE
					v_SP = 0.  # SP
					# v_HD = 0.
					# v_Recall = 0.

					for i, (images, pet,  GT) in enumerate(self.valid_loader):

						images = images.to(self.device)
						pet = pet.to(self.device)
						GT = GT.to(self.device)
						SR, loss_re, loss_score_pet, loss_score_ct, psnr, ssim = self.unet(images, pet, num_modal=1)


						v_TPR += get_TPR(SR, GT)
						v_FPR += get_FPR(SR, GT)
						v_PPV += get_precision(SR, GT)
						v_JS += get_JS(SR, GT)
						v_DC += get_DC(SR, GT)
						v_SE += get_sensitivity(SR, GT)
						v_SP += get_specificity(SR, GT)
						# v_HD += get_HD(SR, GT)
						# v_Recall += get_Recall(SR, GT)

					length = len(self.valid_loader)
					v_TPR = v_TPR / length
					v_FPR = v_FPR / length
					v_PPV = v_PPV / length
					v_JS = v_JS / length
					v_DC = v_DC / length
					v_SE = v_SE / length
					v_SP = v_SP / length
					# v_HD = v_HD / length
					# v_Recall = v_Recall / length

					"""
					self.scheduler.step()
					# learn_rate = self.scheduler.get_lr()
					zdy = self.scheduler.get_lr()
					learn_rate = zdy[0]

					"""
					# Print the log info
					print('[Validation] TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE:%.4f, SP:%.4f\n[Loss]: %.4f  [lr]: %.4f'   # , HD:%.4f, Recall:%.4f
					      % (v_TPR, v_FPR, v_PPV, v_JS, v_DC, v_SE, v_SP, epoch_loss, learn_rate))     # , v_HD, v_Recall

					"""
					torchvision.utils.save_image(images.data.cpu(),
                                                 os.path.join(self.val_result_path,
                                                              '%s_val_%d_image.jpg' % (self.model_type, epoch + 1)))
					torchvision.utils.save_image(SR.data.cpu(),
                                                 os.path.join(self.val_result_path,
                                                              '%s_val_%d_SR.jpg' % (self.model_type, epoch + 1)))
					torchvision.utils.save_image(GT.data.cpu(),
                                                 os.path.join(self.val_result_path,
                                                              '%s_val_%d_GT.jpg' % (self.model_type, epoch + 1)))
                    			"""

					

					h = open(os.path.join(self.val_result_path, 'val_result.csv'), 'a', encoding='utf-8', newline='')
					wr = csv.writer(h)
					wr.writerow([self.model_type, v_TPR, v_FPR, v_PPV, v_JS, v_DC, v_SE, v_SP, epoch + 1, self.num_epochs])    # , v_HD, v_Recall
					h.close()

					ve = open(os.path.join(self.val_result_path, 'valsim_result.csv'), 'a', encoding='utf-8', newline='')
					wx = csv.writer(ve)
					wx.writerow([epoch + 1, psnr, ssim])
					ve.close()

					self.print_gpu_memory_usage()
					
					# Save Best U-Net model
					if v_DC + v_JS > best_Dice:
						best_Dice = v_DC + v_JS
						best_unet = self.unet.state_dict()
						print('Best %s model score : %.4f' % (self.model_type, best_Dice))
						torch.save(best_unet, unet_path)

				if epoch % 99 == 0:
					unet_epoch_100_path = os.path.join(self.model_path, 'unet_epoch-%d.pkl'% (epoch))
					torch.save(self.unet.state_dict(), unet_epoch_100_path)


	# ===================================== Test ====================================#
	def test(self):

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f.pkl' % (self.model_type, self.num_epochs, self.lr))

		self.unet.load_state_dict(torch.load(unet_path))
		print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))


		with torch.no_grad():
			for epoch in range(self.num_epochs_test):

				TPR = 0.  # TPR
				FPR = 0.  # FPR
				PPV = 0.   # PPV
				JS = 0.    # Jaccard Similarity
				DC = 0.    # Dice Coefficient
				SE = 0.
				SP = 0.
				HD = 0.
				Recall = 0.
				for i, (images, pet,  GT) in enumerate(self.test_loader):

					images = images.to(self.device)
					pet = pet.to(self.device)
					GT = GT.to(self.device)
					SR, loss_re, loss_score_pet, loss_score_ct, psnr, ssim = self.unet(images, pet, num_modal=1)
					
					TPR += get_TPR(SR, GT)
					FPR += get_FPR(SR, GT)
					PPV += get_precision(SR, GT)
					JS += get_JS(SR, GT)
					DC += get_DC(SR, GT)
					SE += get_sensitivity(SR, GT)
					SP += get_specificity(SR, GT)
					HD += get_HD(SR, GT)
					Recall += get_Recall(SR, GT)


				length = len(self.test_loader)
				TPR = TPR / length
				FPR = FPR / length
				PPV = PPV / length
				JS = JS/length
				DC = DC/length
				SE = SE / length
				SP = SP / length
				HD = HD / length
				Recall = Recall / length

				print('Epoch [%d/%d], \n[Test] TPR: %.4f, FPR: %.4f, PPV:%.4f, JS: %.4f, DC: %.4f, SE: %.4f, SP: %.4f, HD: : %.4f, Recall: %.4f'%
					  (epoch + 1, self.num_epochs_test, TPR, FPR, PPV, JS, DC,SE, SP, HD, Recall))

				#  Visualization of segmentation results
				images = images * 0.5 + 0.5
				pet = pet * 0.5 + 0.5

				batch_size, _, _, _ = images.shape
				split_tensors1 = torch.chunk(images, batch_size)
				for i, split_tensor in enumerate(split_tensors1):
					torchvision.utils.save_image(split_tensor.data.cpu().squeeze(0), os.path.join(self.test_result_path,
							'%s_test_%d_%d_image.jpg' % (self.model_type, epoch + 1, i)))

				batch_size, _, _, _ = pet.shape
				split_tensors2 = torch.chunk(pet, batch_size)
				for i, split_tensor in enumerate(split_tensors2):
					torchvision.utils.save_image(split_tensor.data.cpu().squeeze(0), os.path.join(self.test_result_path,
							'%s_test_%d_%d_pet.jpg' % (self.model_type, epoch + 1, i)))

				batch_size, _, _, _ = SR.shape
				split_tensors3 = torch.chunk(SR, batch_size)
				for i, split_tensor in enumerate(split_tensors3):
					torchvision.utils.save_image(split_tensor.data.cpu().squeeze(0), os.path.join(self.test_result_path,
							'%s_test_%d_%d_SR.jpg' % (self.model_type, epoch + 1, i)))

				batch_size, _, _, _ = GT.shape
				split_tensors4 = torch.chunk(GT, batch_size)
				for i, split_tensor in enumerate(split_tensors4):
					torchvision.utils.save_image(split_tensor.data.cpu().squeeze(0), os.path.join(self.test_result_path,
							'%s_test_%d_%d_GT.jpg' % (self.model_type, epoch + 1, i)))



				g = open(os.path.join(self.test_result_path,'test_result.csv'), 'a', encoding='utf-8', newline='')
				wr = csv.writer(g)
				wr.writerow([self.model_type,TPR,FPR, PPV, JS, DC, SE, SP, HD, Recall, self.lr, epoch + 1, self.num_epochs_test ])   #,self.augmentation_prob
				g.close()

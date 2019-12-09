import tempfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K

	
class LearningRateFinder:
	def __init__(self, model, stopFactor=4, beta=0.98):
		self.model = model
		self.stopFactor = stopFactor
		self.beta = beta
		
		self.lrs = None
		self.lrMult = None
		self.losses = None
		self.best_loss = None
		self.smooth_loss = None
		self.batch_num = None
		
	def reset(self):
		self.lrs = []
		self.lrMult = 1
		self.losses = []
		self.best_loss = 1e9
		self.smooth_loss = 0
		self.batch_num = 0
		
	def plot(self, start=None, end=None):
		if start is None:
			start = 0
		if end is None:
			end = len(self.losses)

		plt.figure(figsize=(8, 6))
		plt.subplot(2, 1, 1)
		plt.plot(range(start, end), self.losses[start:end])
		plt.ylabel("loss")
		plt.subplot(2, 1, 2)
		plt.plot(range(start, end), self.lrs[start:end])
		plt.ylabel("learning rate")
		plt.show()
		
	def on_batch_end(self, batch, logs):
		"""
		Save learning rate in lrs
		Update learning rate value
		Calculate smooth loss and save in losses
		"""
		# Get current learning rate and save it
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)
		
		# Calculate smooth loss and save it
		loss = logs["loss"]
		self.batch_num += 1
		self.smooth_loss = (self.beta * self.smooth_loss) + ((1 - self.beta) * loss)
		correctedloss = self.smooth_loss / (1 - (self.beta ** self.batch_num))
		self.losses.append(correctedloss)
		
		# Calculate stop loss
		stoploss = self.stopFactor * self.best_loss
		
		if self.batch_num > 1 and self.smooth_loss > stoploss:
			# Stop training
			self.model.stop_training = True
			return
			
		if correctedloss < self.best_loss:
			# Update best loss
			self.best_loss = correctedloss
		
		# Increase learning rate
		lr *= self.lrMult
		K.set_value(self.model.optimizer.lr, lr)
		
	def find(self, training_data, start_lr=1e-10, end_lr=1e+1, batch_size=32, epochs=5, sample_size=None, verbose=1):
		# Reset parameters
		self.reset()
		
		# If sample size is not defined, use length of training data
		if sample_size is None:
			sample_size = len(training_data[0])
			
		# Calculate update rate for learning rate
		updateTimePerEpoch = np.ceil(sample_size / float(batch_size))
		updateTimeTotal = epochs * updateTimePerEpoch
		self.lrMult = (end_lr / start_lr) ** (1.0 / updateTimeTotal)
			
		# Save model weights and learning rate, so we can reset it later
		weightsFile = tempfile.mkstemp()[1]
		self.model.save_weights(weightsFile)
		orig_lr = K.get_value(self.model.optimizer.lr)
		
		# Create callback function to update learning rate every batch
		callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
		
		# Run training
		K.set_value(self.model.optimizer.lr, start_lr)
		self.model.fit(training_data[0], training_data[1],
					   batch_size=batch_size,
					   epochs=epochs,
					   verbose=verbose,
					   callbacks=[callback])
		
		# Load model weights back
		self.model.load_weights(weightsFile)
		K.set_value(self.model.optimizer.lr, orig_lr)
		

if __name__ == "__main__":
	lr_finder = LearningRateFinder(None)

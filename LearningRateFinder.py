import tempfile
import numpy as np
import matplotlib.pyplot as plt


class LearningRateFinder:
	def __init__(self, model, stop_factor=4, beta=0.98):
		self.model = model
		self.stop_factor = stop_factor
		self.beta = beta

		self.lrs = None
		self.lrMult = None
		self.losses = None
		self.best_loss = None
		self.smooth_loss = None
		self.stop_training = None

	def reset(self):
		self.lrs = []
		self.lrMult = 1
		self.losses = []
		self.best_loss = 1e9
		self.smooth_loss = 0
		self.stop_training = False

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

	def on_batch_end(self, batch, loss):
		# Save current learning rate
		lr = self.model.optimizer.lr.numpy()
		self.lrs.append(lr)

		# Save smooth loss
		self.smooth_loss = (self.beta * self.smooth_loss) + ((1 - self.beta) * loss)
		corrected_loss = self.smooth_loss / (1 - (self.beta ** batch))
		self.losses.append(corrected_loss)

		# Calculate stop loss
		stop_loss = self.stop_factor * self.best_loss
		if batch > 0 and corrected_loss > stop_loss:
			# Stop training
			self.stop_training = True
			return

		if corrected_loss < self.best_loss:
			# Update best loss
			self.best_loss = corrected_loss

		# Increase learning rate
		lr *= self.lrMult
		self.model.optimizer.lr = lr

	def find(self, x, y, start_lr=1e-10, end_lr=1e+1, batch_size=32, epochs=5):
		# Reset parameters
		self.reset()

		# Get sample size
		sample_size = x.shape[0]

		# Calculate update rate for learning rate
		updateTimePerEpoch = np.ceil(sample_size / float(batch_size))
		updateTimeTotal = epochs * updateTimePerEpoch
		self.lrMult = (end_lr / start_lr) ** (1.0 / updateTimeTotal)

		# Create Dataset
		train_data = tf.data.Dataset.from_tensor_slices((x, y))
		train_data = train_data.cache()
		train_data = train_data.shuffle(sample_size).batch(batch_size)
		train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

		# Save model weights and learning rate for restoring after train
		weightsFile = tempfile.mkstemp()[1]
		self.model.save_weights(weightsFile)
		orig_lr = self.model.optimizer.lr.numpy()

		# Run training
		for epoch in range(epochs):
			for batch, (x, y) in enumerate(train_data):
				# Forward propagation
				with tf.GradientTape() as g:
					g.watch(x)
					y_pred = self.model(x)
					loss = self.model.loss(y, y_pred)
				# Backpropagation
				grads = g.gradient(loss, self.model.trainable_variables)
				# Update model parameters
				self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

				# Update Learning rate
				self.on_batch_end(batch + 1, loss)

				# Stop training
				if self.stop_training:
					# Load model weights back
					self.model.load_weights(weightsFile)
					self.model.optimizer.lr = orig_lr
					return
			# Report
			print('Epoch {} Loss {:.4f} LearningRate {:.4f}'.format(epoch + 1, loss, self.model.optimizer.lr.numpy()))

		# Load model weights back
		self.model.load_weights(weightsFile)
		self.model.optimizer.lr = orig_lr
		

if __name__ == "__main__":
	lr_finder = LearningRateFinder(None)

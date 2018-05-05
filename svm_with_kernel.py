
'''
# ----------------------------------------------------------------------------------
# Project : Implementation of hard SVM and soft SVM, polynomial, gaussian kernels for SVM
# Course  : EE 769 Introduction to Machine Learning.
# Sources :
# 1. for optimization, library "CVXOPT" from cvxopt.org and concepts of optimazition
#    from "convex optimization by Boyd" & lecture notes.
# 2. for understanding SVM and kernels, lecture notes.
# 3. youtube videos to generate random data sets.
# -----------------------------------------------------------------------------------
'''
#print(__doc__)
import numpy as np
from numpy import linalg

# for quadratic programming
import cvxopt
import cvxopt.solvers



# The linear kernel in SVM is implemented as dot product of the features x_i, x_j
# That is done using numpy's dot product library functions
def linear_kernel(x1, x2):
	return np.dot(x1, x2)


# For degree-d polynomials, the polynomial kernel is defined as[2]
#     K( x_i, x_j ) = (<x_i,x_j>  + c )^d
# where x_i and x_j are vectors in the input space, i.e. vectors of features computed from training or test samples
# and c >= 0 is a free parameter trading off the influence of higher-order versus lower-order terms in the polynomial.
def polynomial_kernel(x, y, p=3):
	return (1 + np.dot(x, y)) ** p

# gaussian kernel K( x_i, x_j ) = exp{(-1/sigma^2)(||x_i - x_j||^2)}

def gaussian_kernel(x, y, sigma=5.0):
	return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class classifier_SVM(object):

	def __init__(self, kernel=linear_kernel, C=None):
		self.kernel = kernel
		self.C = C
		if self.C is not None: self.C = float(self.C)

	# for training
	def fit(self, X, y):
		n_samples, n_features = X.shape

		# Gram matrix
		# In linear algebra, the Gram matrix (Gramian matrix or Gramian)
		# of a set of vectors x_1, ... , x_n in an inner product space is the Hermitian matrix of inner products,
		# whose entries are given by G_ij = < x_i , x_j >

		K = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				K[i,j] = self.kernel(X[i], X[j])

		#Solves the pair of primal and dual convex quadratic programs
		# min J = (1/2) x^TPx + q^T x \\ subject to Gx <= h , Ax = b
		# obj function is dual of general SVM obj fun with kernels
		P = cvxopt.matrix(np.outer(y,y) * K)
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y, (1,n_samples))
		b = cvxopt.matrix(0.0)

		if self.C is None:
			G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
			h = cvxopt.matrix(np.zeros(n_samples))
		else:
			tmp1 = np.diag(np.ones(n_samples) * -1)
			tmp2 = np.identity(n_samples)
			G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
			tmp1 = np.zeros(n_samples)
			tmp2 = np.ones(n_samples) * self.C
			h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

		# solve QP problem
		solution = cvxopt.solvers.qp(P, q, G, h, A, b)

		# Lagrange multipliers 'a'(alphas)
		a = np.ravel(solution['x'])

		# Support vectors have non zero lagrange multipliers
		# alphas will always be positive acc to obj fun
		sv = a > 1e-5
		ind = np.arange(len(a))[sv]
		self.a = a[sv]
		self.sv = X[sv]
		self.sv_y = y[sv]
		# print("%d support vectors out of %d points" % (len(self.a), n_samples))

		# Intercept or bias term 'b'
		# for intercept or bias update check "understanding ML theory" by shai ben david
		self.b = 0
		for n in range(len(self.a)):
			self.b += self.sv_y[n]
			self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
		self.b /= len(self.a)

		# Weight vector
		# for weight update check "understanding ML theory" by shai ben david
		if self.kernel == linear_kernel:
			self.w = np.zeros(n_features)
			for n in range(len(self.a)):
				self.w += self.a[n] * self.sv_y[n] * self.sv[n]
		else:
			# update of weights depends on kernel functions
			# hence weights updated in project()
			self.w = None

	def project(self, X):
		# y_pred = signum(w'X+b)
		# project() will project on to the space of 'weights'
		if self.w is not None:
			return np.dot(X, self.w) + self.b
		else:
			y_predict = np.zeros(len(X))
			for i in range(len(X)):
				s = 0
				for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
					s += a * sv_y * self.kernel(X[i], sv)
				y_predict[i] = s
			return y_predict + self.b

	def predict(self, X):
		# y_pred = signum(w'X+b)
		# project() will project on to the space of 'weights'
		return np.sign(self.project(X))

if __name__ == "__main__":
	import pylab as pl

	def lin_separable_data_gen():
		# generate training data in the 2-d case
		# to generste linear separable data for hard svm test
		mean1 = np.array([0, 2])
		mean2 = np.array([2, 0])
		cov = np.array([[0.8, 0.6], [0.6, 0.8]])
		X1 = np.random.multivariate_normal(mean1, cov, 100)
		y1 = np.ones(len(X1))
		X2 = np.random.multivariate_normal(mean2, cov, 100)
		y2 = np.ones(len(X2)) * -1
		return X1, y1, X2, y2

	def nonlin_separable_data_gen():
		
		# generate non-linearly separable data for testing of polynomial 
		# and gaussian kernel SVM in 2d case
		mean1 = [-1, 2]
		mean2 = [1, -1]
		mean3 = [4, -4]
		mean4 = [-4, 4]
		cov = [[1.0,0.8], [0.8, 1.0]]
		X1 = np.random.multivariate_normal(mean1, cov, 50)
		X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
		y1 = np.ones(len(X1))* -1
		X2 = np.random.multivariate_normal(mean2, cov, 50)
		X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
		y2 = np.ones(len(X2))
		return X1, y1, X2, y2

	def overlapdata_gen():
		# generate overlapped training data in the 2-d case
		# to test soft svm
		mean1 = np.array([0, 2])
		mean2 = np.array([2, 0])
		cov = np.array([[2.5, 1.0], [1.0, 2.5]])
		X1 = np.random.multivariate_normal(mean1, cov, 100)
		y1 = np.ones(len(X1))
		X2 = np.random.multivariate_normal(mean2, cov, 100)
		y2 = np.ones(len(X2)) * -1
		return X1, y1, X2, y2

	def training_data_split(X1, y1, X2, y2):
		# splitting the data in 90:10 ratio for training
		X1_train = X1[:90]
		y1_train = y1[:90]
		X2_train = X2[:90]
		y2_train = y2[:90]
		X_train = np.vstack((X1_train, X2_train))
		y_train = np.hstack((y1_train, y2_train))
		return X_train, y_train

	def test_data_split(X1, y1, X2, y2):
		X1_test = X1[90:]
		y1_test = y1[90:]
		X2_test = X2[90:]
		y2_test = y2[90:]
		X_test = np.vstack((X1_test, X2_test))
		y_test = np.hstack((y1_test, y2_test))
		return X_test, y_test

	def plot_margin(X1_train, X2_train, clf , title_):
		def f(x, w, b, c=0):
			# given x, return y such that [x,y] in on the line
			# w.x + b = c
			return (-w[0] * x - b + c) / w[1]

		pl.plot(X1_train[:,0], X1_train[:,1], "ro")
		pl.plot(X2_train[:,0], X2_train[:,1], "bo")
		#pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

		# w.x + b = 0
		a0 = -4; a1 = f(a0, clf.w, clf.b)
		b0 = 4; b1 = f(b0, clf.w, clf.b)
		pl.plot([a0,b0], [a1,b1], "k")

		# w.x + b = 1
		a0 = -4; a1 = f(a0, clf.w, clf.b, 1)
		b0 = 4; b1 = f(b0, clf.w, clf.b, 1)
		pl.plot([a0,b0], [a1,b1], "k--")

		# w.x + b = -1
		a0 = -4; a1 = f(a0, clf.w, clf.b, -1)
		b0 = 4; b1 = f(b0, clf.w, clf.b, -1)
		pl.plot([a0,b0], [a1,b1], "k--")

		pl.xlabel('X1')
		pl.ylabel('X2')
		pl.title(title_)
		pl.axis("tight")
		pl.show()

	def plot_contour(X1_train, X2_train, clf , title_):
		pl.plot(X1_train[:,0], X1_train[:,1], "ro")
		pl.plot(X2_train[:,0], X2_train[:,1], "bo")
		#pl.scatter(clf.sv[:,0], clf.sv[:,1], s = 100, c = "g")

		X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
		X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
		Z = clf.project(X).reshape(X1.shape)
		pl.contour(X1, X2, Z, [0.0], colors ='k', linewidths = 1, origin = 'lower')
		pl.contour(X1, X2, Z + 1, [0.0], colors='grey' , linewidths=1 , origin ='lower')
		pl.contour(X1, X2, Z - 1, [0.0], colors='grey' , linewidths=1 , origin ='lower')

		pl.xlabel('X1')
		pl.ylabel('X2')
		pl.title(title_)
		pl.axis("tight")
		pl.show()

	#------------------------------for Hard SVM-----------------------------

	def test_linear():
		X1, y1, X2, y2 = lin_separable_data_gen()
		X_train, y_train = training_data_split(X1, y1, X2, y2)
		X_test, y_test = test_data_split(X1, y1, X2, y2)

		clf = classifier_SVM()
		clf.fit(X_train, y_train)

		y_predict = clf.predict(X_test)
		correct = np.sum(y_predict == y_test)
		#acc = correct/len(y_predict)*100
		print("%d out of %d predictions correct" % (correct, len(y_predict) ))

		plot_margin(X_train[y_train == 1], X_train[y_train == -1], clf , title_ = 'Hard SVM')
		#plot_margin(X_test[y_test == 1], X_test[y_test == -1], clf , title_ = 'Hard SVM')

	#---------------------------------for polynomial kernel---------------------

	def test_non_linear():
		X1, y1, X2, y2 = nonlin_separable_data_gen()
		X_train, y_train = training_data_split(X1, y1, X2, y2)
		X_test, y_test = test_data_split(X1, y1, X2, y2)

		clf = classifier_SVM(polynomial_kernel)
		clf.fit(X_train, y_train)

		y_predict = clf.predict(X_test)
		correct = np.sum(y_predict == y_test)
		#acc = correct/len(y_predict)*100
		print("%d out of %d predictions correct" % (correct, len(y_predict)))
		title_ = 'polynomial kernel with d = 3'
		plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf ,title_)


	# ----------------------for soft SVM---------------------------------------
	def test_soft():
		X1, y1, X2, y2 = overlapdata_gen()
		X_train, y_train = training_data_split(X1, y1, X2, y2)
		X_test, y_test = test_data_split(X1, y1, X2, y2)

		clf = classifier_SVM(C = 1000.1)
		clf.fit(X_train, y_train)

		y_predict = clf.predict(X_test)
		correct = np.sum(y_predict == y_test)
		#acc = correct/len(y_predict)*100
		print("%f out of %f predictions correct" % (correct, len(y_predict)))

		title_ = 'Soft SVM'
		plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf , title_)



	# ---------------for gaussian kernels--------------------------------------
	def test_gaussian_nonlinear():
		X1, y1, X2, y2 = nonlin_separable_data_gen()
		X_train, y_train = training_data_split(X1, y1, X2, y2)
		X_test, y_test = test_data_split(X1, y1, X2, y2)

		clf = classifier_SVM(gaussian_kernel)
		clf.fit(X_train, y_train)

		y_predict = clf.predict(X_test)
		correct = np.sum(y_predict == y_test)
		#acc = correct/len(y_predict)*100
		print("%d out of %d predictions correct" %(correct, len(y_predict)))


		title_ = 'RBF kernel with sigma = 5.0'
		plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf , title_)


	# for Soft SvM testing uncomment the following
	#print("\nSoft SVM performing on the Linear separable Gaussian overlapped data generated\n")
	#test_soft()

	#for Hard SVM test uncomment the following
	#print("\nLinear kernel performing on the Linear separable Gaussian data generated\n")
	#test_linear()

	# for testing of polynomial kernel SVM uncomment the following
	#print("\nNonLinear Polynomial kernel of order : 3 performing on the Gaussian data generated\n")
	#test_non_linear()

	# for testing of gaussian kernel SVM uncomment the following
	print("\nGaussian Kernel\n")
	test_gaussian_nonlinear()

	#--------------------------------------------------------------------------------------
	#----------------------------THE END---------------------------------------------------
	#--------------------------------------------------------------------------------------

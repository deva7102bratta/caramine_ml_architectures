import numpy as np
import pickle

class LogisticRegression:
  def __init__(
    self,
    input_size,
    learning_rate=0.01,
    epochs=300,
    batch_size=None,
    threshold=0.5,
    fit_intercept=True,
    shuffle=True,
    random_state=None
  ):
    if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size<=0:
      raise ValueError("Input_size must be positive integer")

    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0:
      raise ValueError("learning_rate must be a positive finite number.")

    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
      raise ValueError("epochs must be a positive integer.")

    if batch_size is not None:
      if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(threshold, (float, int)) or isinstance(threshold, bool) or  not 0 < threshold < 1:
      raise ValueError("threshold must be between 0 and 1")

    if not isinstance(fit_intercept, bool):
      raise ValueError("fit_intercept must be a boolean")

    if not isinstance(shuffle, bool):
      raise ValueError("shuffle must be a boolean")

    if random_state is not None:
      if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer or None")

    self.input_size = input_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
    self.threshold = threshold
    self.fit_intercept = fit_intercept
    self.shuffle = shuffle

    self.rng = np.random.default_rng(random_state)

    # One weight per input feature
    self.weights = np.zeros(input_size)

    # One shared bias
    self.bias = 0.0

    self.is_fitted = False
    self.history = {"loss": []}

  def _sigmoid(self, z):
    z = np.clip(z, -500, 500)
    return 1/(1+np.exp(-z))

  def _validate_X(self, X):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
      X = X.reshape(1, -1)

    if X.ndim != 2:
      raise ValueError("X must be a 2D array.")

    if X.shape[1] != self.input_size:
      raise ValueError(
        f"Expeceted {self.input_size} features, "
        f"Received {X.shape[1]}."
      )

    if X.shape[0] == 0:
      raise ValueError("X cannot be empty.")

    if not np.all(np.isfinite(X)) :
      raise ValueError("X must contain finite values.")

    return X

  def _validate_y(self, y, n_samples):
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(y) != n_samples:
      raise ValueError("X and y must have same sample count.")

    if not np.all(np.isin(y, [0, 1])):
      raise ValueError("y must contain only 0 and 1.")

    if np.unique(y).size != 2:
      raise ValueError("Training data must contain both classes.")

    return y

  def _forward(self, X):
    z = X @ self.weights

    if self.fit_intercept:
      z = z + self.bias

    return self._sigmoid(z)

  def _loss(self, y, probabilities):
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1-eps)

    return -np.mean(
      y*np.log(probabilities)
      + (1-y)*np.log(1-probabilities)
    )

  def fit(self, X, y):
    X = self._validate_X(X)
    n_samples = X.shape[0]
    y = self._validate_y(y, n_samples)


    for epoch in  range(self.epochs):

      if self.shuffle:
        indices = self.rng.permutation(n_samples)
        X_epoch = X[indices]
        y_epoch = y[indices]
      else:
        X_epoch = X
        y_epoch = y

      size = self.batch_size or n_samples

      for start in range(0, n_samples, size):
        X_batch = X_epoch[start:start+size]
        y_batch = y_epoch[start:start+size]

        probabilities = self._forward(X_batch)
        error = probabilities - y_batch

        dw = X_batch.T @ error / len(y_batch)
        db = np.mean(error)

        self.weights -= self.learning_rate*dw

        if self.fit_intercept:
          self.bias -= self.learning_rate * db

      # Record loss after each epoch
      probabilities = self._forward(X)
      current_loss = self._loss(y, probabilities)
      self.history["loss"].append(current_loss)

    self.is_fitted = True
    return self

  def decision_function(self, X):
    self._check_fitted()
    X = self._validate_X(X)

    z = X @ self.weights

    if self.fit_intercept():
      z = z+self.bias

    return z

  def predict_proba(self, X):
    return self._sigmoid(self.decision_function(X))

  def predict(self, X, threshold=None):
    threshold = self.threshold if thresholdis is None else threshold

    if not 0 < threshold < 1 :
      raise ValueError("threshold must be between 0 and 1")

    probabilities = self.predict_proba(X)
    return (probabilities >= threshold).astype(int)

  def get_params(self):
    return {
      "input_size": self.input_size,
      "learning_rate": self.learning_rate,
      "epochs": self.epochs,
      "batch_size": self.batch_size,
      "threshold": self.threshold,
      "fit_intercept": self.fit_intercept,
      "shuffle": self.shuffle
    }

  def get_weights(self):
    return self.weights.copy()

  def get_bias(self):
    return self.bias

  def get_history(self):
    return {
      key: values.copy()
      for key, values in self.history.items()
    }

  def save(self, path):
    self._check_fitted()

    state = {
      "input_size": self.input_size,
      "learning_rate": self.learning_rate,
      "epochs": epochs,
      "batch_size": self.batch_size,
      "threshold": self.threshold,
      "fit_intercept": self.fit_intercept,
      "shuffle": self.shuffle,
      "weights": self.weights,
      "bias": self.bias,
      "history": self.history,
      "is_fitted": self.is_fitted
    }

    with open(path, "wb") as file:
      pickle.dump(state, file)

  @classmethod
  def load(cls, path):
    with open(path, "rb") as file:
      data = pickle.load(file)

    model = cls(
      input_size=data["input_size"],
      learning_rate=data["learning_rate"],
      epochs=data["epochs"],
      batch_size=data["batch_size"],
      threshold=data["threshold"],
      fit_intercept=data["fit_intercept"],
      shuffle=data["shuffle"],
    }

    weights = np.asarray(data["weights"], dtype=float)
    bias = data["bias"]

    if weights.shape != (model.input_size,):
      raise ValueError("Invalid coefficient dimensions.")

    if not np.all(np.isfinite(weights)):
      raise ValueError("Model coefficients must be finite.")

    if isinstance(bias, bool) or not isinstance(weights, (int, float)):
      raise ValueError("Invalid intercept.")

    if not np.isfinite(bias):
      raise ValueError("Model intercept must be finite.")

    model.weights = weights
    model.bias = float(bias)
    model.history = data["history"]
    model.is_fitted_ = True

    return model

  def _check_fitted(self):
    if not self.is_fitted:
      raise RuntimeError("Train or load the model first.")

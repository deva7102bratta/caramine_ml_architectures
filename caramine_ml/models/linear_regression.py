import json
import numpy as np

class LinearRegression:

  VALID_OPTIMIZERS = {
    "batch_gradient_descent",
    "mini_batch_gradient_descent",
    "stochastic_gradient_descent",
  }

  def __init__(
    self,
    input_size,
    learning_rate=0.001,
    epochs=300,
    batch_size=32,
    optimizer="mini_batch_gradient_descent"
  ):
    # Validate configuration
    if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size<=0:
      raise ValueError("input_size must be positive integer.")

    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0:
      raise ValueError("learning_rate must be positive and finite.")

    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <=0:
      raise ValueError("epochs must be a positive integer.")

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <=0:
      raise ValueError("batch_size must be a positive integer.")

    if optimizer not in self.VALID_OPTIMIZERS:
      raise ValueError(
        f"optimizer must be one of {sorted(self.VALID_OPTIMIZERS)}"
      )

    self.input_size = input_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
    self.optimizer = optimizer

    # Model Parameters
    self.coef_ = np.zeros(input_size, dtype=float)
    self.intercept_ = 0.0

    # Training state 
    self.is_fitted_ = False
    self.loss_history_ = []

  def _validate_X(self, X):
    # Convert input features to a valid NumPy array.

    try:
      X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
      raise ValueError("X must contain numerical values.") from exc

    # Allow a single sample during prediction.
    if X.ndim == 1:
      X = X.reshape(1, -1)

    if X.ndim !=2:
      raise ValueError("X must be a two-dimensional array.")

    if X.shape[1] != self.input_size:
      raise ValueError(
        f"Expected {self.input_size} features, "
        f"but received {X.shape[1]}"
      )

    if X.shape[0] == 0:
      raise ValueError("X cannot be empty.")

    if not np.all(np.isfinite(X)):
      raise ValueError("X must contain only finite values.")

    return X

  def _validate_y(self, y, n_samples):
    # Validate target values.

    try:
      y = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
      raise ValueError("y must contain numerical values.") from exc

    if y.ndim == 2 and y.shape[1] == 1:
      y = y.ravel()

    if y.ndim != 1:
      raise ValueError("y must be one-dimensional.")

    if len(y) != n_samples:
      raise ValueError("Y must contain the same number of samples.")

    if len(y) == 0:
      raise ValueError("y cannot be empty.")

    if not np.all(np.isfinite(y)):
      raise ValueError("y must contain only finite values.")

    return y

  def _predict_raw(self, X):
    # Calculate predictions without public input validation.

    return X @ self.coef_ + self.intercept_

  def _update_parameters(self, X_batch, y_batch):
    # Calculate gradients and update coefficients and intercept.

    n = len(y_batch)

    # Forward pass
    predictions = self._predict_raw(X_batch)

    # Predictions errors
    errors = predictions - y_batch

    # Gradient of mean squared error
    grad_coef = (2/n)*(X_batch.T @ errors)
    grad_intercept = 2 * np.mean(errors)

    # Gradient descent update
    self.coef_ -= self.learning_rate * grad_coef
    self.intercept_ -= self.learning_rate * grad_intercept

  def fit(self, X, y):
    # Train the model using the selected optimizer

    X = self._validate_X(X)

    # During training require X to be a proper matrix.
    if X.shape[0] == 0:
      raise ValueError("Training data cannot be empty.")

    y = self._validate_y(y, len(X))

    # Initialize parameters only for a new training run.
    if not self.is_fitted_:
      self.coef_ = np.zeros(self.input_size, dtype=float)
      self.intercept_ = 0.0
      self.loss_history_ = []

    n_samples = len(X)

    for epoch in range(self.epochs):

      if self.optimizer == "batch_gradient_descent":

        # One update using the entire training dataset.
        self._update_parameters(X, y)

      elif self.optimizer == "mini_batch_gradient_descent":

        # Shuffle the samples before forming mini-batches.
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, self.batch_size):
          batch_indices = indices[start: start + self.batch_size]

          X_batch = X[batch_indices]
          y_batch = y[batch_indices]

          self._update_parameters(X_batch, y_batch)

      elif self.optimizer == "stochastic_gradient_descent":

        # One update pe sample, in shuffled order.
        indices = np.random.permutation(n_samples)

        for i in indices:
          self._update_parameters(
            X[i:i+1],
            y[i:i+1]
          )

      # Measure the full-dataset loss after this epoch
      predictions = self._predict_raw(X)
      loss = np.mean((predictions - y)**2)

      self.loss_history_.append(float(loss))

      if not np.isfinite(loss):
        raise FloatingPointError("Training diverged. Try a smaller learning_rate")

    self.is_fitted_ = True
    return self

  def predict(self, X):
    "Predict target values for new input samples"

    if  not self.is_fitted_:
      raise RuntimeError("Train the model using fit() before predict().")

    X = self._validate_X(X)

    predictions = self._predict_raw(X)

    if not np.all(np.isfinite(predictions)):
      raise FloatingPointError("Predictions are not finite.")

    return predictions

  def score(self, X, y):
    "Calculate the coefficients of setermination, R^2."

    X = self._validate_X(X)
    y = self._validate_y(y, len(X))

    predictions = self.predict(X)

    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
      return 1.0 if ss_res == 0 else 0.0
    return float(1-ss_res/ss_tot)

  def save(self, path):
    # Save the trained model and its configuration as JSON

    if not self.is_fitted_:
      raise RuntimeError("Train the model before saving")

    data = {
      "input_size": self.input_size,
      "learning_rate": self.learning_rate,
      "epochs": self.epochs,
      "batch_size": self.batch_size,
      "optimizer": self.optimizer,
      "coef": self.coef_.tolist(),
      "intercept": float(self.intercept_)
    }

    with open(path, "w", encoding="utf-8") as file:
      json.dump(data, file, indent=4)

  @classmethod
  def load_model(cls, path):
    #Load a trained model from a JSON file.

    with open(path, "r", encoding="utf-8") as file:
      data = json.load(file)

    required = {
      "input_size",
      "learning_rate",
      "epochs",
      "batch_size",
      "optimizer",
      "coef",
      "intercept"
    }

    if not isinstance(data, dict) or not required.issubset(data):
      raise ValueError("Invalid model file.")

    model = cls(
      input_size=data["input_size"],
      learning_rate=data["learning_rate"],
      epochs=data["epochs"],
      batch_size=data["batch_size"],
      optimizer=data["optimizer"]
    )
    coef = np.asarray(data["coef"], dtype=float)
    intercept = data["intercept"]

    if coef.shape != (model.input_size,):
      raise ValueError("Invalid coefficient dimensions.")

    if not np.all(np.isfinite(coef)):
      raise ValueError("Model coefficients must be finite.")

    if isinstance(intercept, bool) or not isinstance(intercept, (int, float)):
      raise ValueError("Invalid intercept.")

    if not np.isfinite(intercept):
      raise ValueError("Model intercept must be finite.")

    model.coef_ = coef
    model.intercept_ = float(intercept)
    model.is_fitted_ = True

    return model


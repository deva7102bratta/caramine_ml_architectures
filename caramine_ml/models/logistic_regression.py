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
    multi_class=False
  ):

    # Validating input_size
    if (
      not isinstance(input_size, (int, np.integer)) 
      or isinstance(input_size, (bool, np.bool_)) 
      or input_size<=0
    ):
      raise ValueError("Input_size must be positive integer")

    # Validate learning_rate
    if (
      not isinstance(learning_rate, (int, float, np.integer, np.floating))
      or isinstance(learning_rate, (bool, np.bool_))
      or not np.isfinite(learning_rate)
      or learning_rate <= 0
    ):
      raise ValueError("learning_rate must be a positive finite number.")

    # Validate epochs
    if (
      not isinstance(epochs, (int, np.integer))
      or isinstance(epochs, (bool, np.bool_))
      or epochs <= 0
    ):
      raise ValueError("epochs must be a positive integer.")

    # Validate batch_size
    if batch_size is not None:
      if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, (bool, np.bool_))
        or batch_size <= 0
      ):
        raise ValueError("batch_size must be a positive integer.")

    # Validate threshold
    if (
      not isinstance(threshold, (float, int, np.integer, np.floating))
      or isinstance(threshold, (bool, np.bool_))
      or not np.isfinite(threshold)
      or  not 0 < threshold < 1
    ):
      raise ValueError("threshold must be between 0 and 1")

    if not isinstance(fit_intercept, (bool, np.bool_)):
      raise ValueError("fit_intercept must be a boolean.")

    if not isinstance(shuffle, (bool, np.bool_)):
      raise ValueError("shuffle must be a boolean.")

    if not isinstance(multi_class, (bool, np.bool_)):
      raise ValueError("multi_class must be a boolean.")

    if random_state is not None:
      if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, (bool, np.bool_)):
        raise ValueError("random_state must be an integer or None")

    self.input_size = int(input_size)
    self.learning_rate = float(learning_rate)
    self.epochs = int(epochs)
    self.batch_size = None if batch_size == None else int(batch_size)
    self.threshold = float(threshold)
    self.fit_intercept = bool(fit_intercept)
    self.shuffle = bool(shuffle)
    self.random_state = None if random_state == None else int(random_state)
    self.multi_class = bool(multi_class)

    self.rng = np.random.default_rng(random_state)

    # Binary mode starts with one weight per feature
    # Multiclass mode initializes the weights after seeing y.
    if self.multi_class:
      self.weights = np.zeros((self.input_size, 0))
      self.bias = np.zeros(0)
    else:
      self.weights = np.zeros(self.input_size)
      self.bias = 0.0

    self.classes_ = None
    self.is_fitted = False
    self.history = {"loss": []}

  # ------------------------------
  # ACTIVATION FUNTIONS
  # -------------------------------

  @staticmethod
  def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0/(1.0 + np.exp(-z))

  @staticmethod
  def _softmax(z):
    # Subtracting the maximum prevents exponential overflow.
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)

    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

  # -------------------------------------------
  # INPUT VALIDATION
  # -------------------------------------------

  def _validate_X(self, X):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
      X = X.reshape(1, -1)

    if X.ndim != 2:
      raise ValueError("X must be a 1D or 2D array.")

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
    y = np.asarray(y).reshape(-1)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if len(y) != n_samples:
        raise ValueError("X and y must have same sample count.")

    if y.dtype.kind not in "biufUS":
        raise ValueError(
            "y must contain numeric or string class labels."
        )

    if y.dtype.kind in "f":
        if not np.all(np.isfinite(y)):
            raise ValueError("y cannot contain NaN or infinity.")

    if y.dtype.kind in "biuf":
        if not np.all(y == np.floor(y)):
            raise ValueError("Numeric class labels must be integers.")

    if self.multi_class:
        classes = np.unique(y)

        if len(classes) < 2:
            raise ValueError(
                "Multiclass training requires at least two classes."
            )

        return y, classes

    if not np.all(np.isin(y, [0, 1])):
        raise ValueError(
            "Binary classification requires labels 0 and 1."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Binary training data must contain both classes."
        )

    return y.astype(int), np.array([0, 1])

  # -----------------------------------------
  # FORWARD PASS
  # ------------------------------------------

  def _forward(self, X):
    z = X @ self.weights

    if self.fit_intercept:
      z = z + self.bias

    if self.multi_class:
      return self._softmax(z)

    return self._sigmoid(z)

  # ---------------------------------------
  # LOSS
  # ---------------------------------------

  def _loss(self, y, probabilities):

    eps = 1e-15
    n_samples = len(y)

    if self.multi_class:

      # Map class labels to probability-column indices.
      class_indices = np.searchsorted(self.classes_, y)

      selected_probabilities = probabilities[
        np.arange(n_samples),
        class_indices
      ]

      selected_probabilities = np.clip(
        selected_probabilities,
        eps,
        1.0
      )

      return float(-np.mean(np.log(selected_probabilities)))

    probabilities = np.clip(probabilities, eps, 1.0-eps)

    return float(-np.mean(y*np.log(probabilities) + (1-y)*np.log(1-probabilities)))

  # ---------------------------------------
  # TRAINING
  # ---------------------------------------
  def fit(self, X, y):

    X = self._validaate_X(X)

    n_samples = X.shape[0]
    y, classes = self._validate_y(y, n_samples)

    # Initialize parameters for the selected mode.
    if self.multi_class:
      n_classes = len(classes)

      self.weights = np.zeros((self.input_size, n_classes), dtype=float)
      self.bias = np.zeros(n_classes, dtype=float)

      # Convert arbitray integer labels into column indices.
      y_encoded = np.searchsorted(classes, y)

    else:
      self.weights = np.zeros(self.input_size, dtype=float)
      self.bias = 0.0

      y_encoded = y

    batch_size = self.batch_size or n_samples
    self.history = {"loss": []}

    for epoch in range(self.epochs):

      if self.shuffle:
        indices = self.rng.permutation(n_samples)

        X_epoch = X[indices]
        y_epoch = y_encoded[indices]

      else:
        X_epoch = X
        y_epoch = y_encoded

      for start in range(0, n_samples, batch_size):

        end = start + batch_size

        X_batch = X_batch[start:end]
        y_batch = y_batch[start:end]

        probabilities = self._forward(X_batch)

        if self.multi_class:

          # Convert class indices to one-hot labels
          Y = np.zeros_like(probabilities)

          Y[np.arange(len(y_batch)),y_batch] = 1.0

          error = probabilities - y_batch

        else:
          error = probabilities - y_batch

        # Gradient of the cross-entropy loss
        dw = X_batch.T @ error / len(X_batch)

        # Update weights
        self.weights -= self.learning_rate * dw

        # Update bias only when intercept is enabled
        if self.fit_intercept:

          db = np.mean(error, axis=0)
          self.bias -= self.learning_rate*db

      # Record loss after each epoch.
      probabilities = self._forward(X)

      current_loss = self._loss(y, probabilities)

      self.history["loss"].append(current_loss)

    self.is_fitted = True
    return self

  # ---------------------------------------------
  # DECISION FUNCTION
  # ----------------------------------------------
  def decision_funtion(self, X):
    self._check_fitted()

    X = self._validaate_X(X)

    z = X @ self.weights

    if self.fit_intercept:
      z = z + self.bias

    return z if self.multi_class else z.ravel()

  # --------------------------------------------
  # PROBABILITY PREDICTION
  # --------------------------------------------
  def predict_proba(self, X):
    
    self._check_fitted()

    X = self._validate_X(X)

    return self._forward(X)

  # ---------------------------------------------
  # CLASS PREDICTION
  # ---------------------------------------------

  def predict(self, X, threshold=None):

    if self.multi_class:

      probabilities = self.predict_proba(X)

      indices = np.argmax(probabilities, axis=1)

      return self.classes_[indices]

    if threshold is None:
      threshold = self.threshold

    if(
      not isinstance(threshold, (int, float, np.integer, np.floating))
      or isinstance(threshold, (bool, np.bool_))
      or not np.isfinite(threshold)
      or not 0 < threshold < 1
    ):
      raise ValueError("threshold must be between 0 and 1.")

    probabilities = self.predict_proba(X)

    return (
      probabilities >= threshold
    ).astype(int)

  # ------------------------------------------------
  # EVALUATION
  # ------------------------------------------------

  def loss(self, X, y):

    self._check_fitted()

    X = self._validaate_X(X)

    y, _ = self._validate_y_for_evaluation(y, len(X))

    probabilities = self._forward(X)

    return self._loss(y, probabilities)

  def _validate_y_for_evaluation(self, y, n_samples):

    y = np.asarray(y).reshape(-1)

    if y.ndim != 1 or len(y) != n_samples:
        raise ValueError("y must contain one label per input example.")

    if y.dtype.kind not in "biufUS":
        raise ValueError(
            "y must contain numeric or string class labels."
        )

    if y.dtype.kind in "f":
        if not np.all(np.isfinite(y)):
            raise ValueError("y cannot contain NaN or infinity.")

    if y.dtype.kind in "biuf":
        if not np.all(y == np.floor(y)):
            raise ValueError("Numeric class labels must be integers.")

    if self.multi_class:

      if not np.all(np.isin(y, self.classes_)):
        raise ValueError("y contains an unknown class label.")

    else:

      if not np.all(np.isin(y, [0, 1])):
        raise ValueError("Binary labels must be 0 or 1.")

    return y, None

  def score(self, X, y):

    predictions = self.predict(X)

    y = np.asarray(y).reshape(-1)

    if len(predictions) != len(y):
      raise ValueError("X and y must have the same sample count.")

    return float(np.mean(predictions == y))

  # ------------------------------------------------
  # MODEL INFORMATION
  # ------------------------------------------------
  def get_params(self):
    return {
      "input_size": self.input_size,
      "learning_rate": self.learning_rate,
      "epochs": self.epochs,
      "batch_size": self.batch_size,
      "threshold": self.threshold,
      "fit_intercept": self.fit_intercept,
      "shuffle": self.shuffle,
      "random_state": self.random_state,
      "multi_class": self.multi_class
    }

  def get_weights(self):
    self._check_fitted()
    return self.weights.copy()

  def get_bias(self):
    self._check_fitted()
    return self.bias.copy()

  def get_history(self):
    return {
      key: values.copy()
      for key, values in self.history.items()
    }

  # -----------------------------------------
  # SAVE
  # -----------------------------------------
  def save(self, path):

    self._check_fitted()

    state = {
      "params": self.get_params(),
      "weights": self.weights,
      "bias": self.bias,
      "classes": self.classes_,
      "history": self.history
    }

    with open(path, "wb") as file:
      pickle.dump(state, file)

  # ---------------------------------------
  # LOAD
  # ---------------------------------------

  @classmethod
  def load_model(cls, path):
    with open(path, "rb") as file:
      data = pickle.load(file)

    model = cls(**data["params"])

    weights = np.asarray(data["weights"], dtype=float)
    bias = data["bias"]
    classes = np.asarray(data["classes"])

    if not np.all(np.isfinite(weights)):
      raise ValueError("Model weight must be finite.")

    if not np.all(np.isfinite(bias)):
      raise ValueError("Model bias must be finite.")

    if model.multi_class:

      n_classes = len(classes)

      if weight.shape != (model.input_size, n_classes):
        raise ValueError("Invalid multiclass weight dimensions.")

      if np.asarray(bias).shape != (n_classes,):
        raise ValueError("Invalid multiclass bias dimensions.")

      if n_classes < 2 or len(np.unique(classes)) != n_classes:
        raise ValueError("Invalid multiclass labels.")

    else:
      if weights.shape != (model.input_size, ):
        raise valueError("Invalid binary weight dimensions.")

      if np.asarray(bias).shape != ():
        raise valueError("Invalid binary bias dimensions.")

      if not np.array_equal(classes, np.array([0, 1])):
        raise valueError("Invalid binary class labels.")

    model.weights = weights
    model.bias = bias.copy() if isinstance(bias, np.ndarray) else float(bias)
    model.classes_ = classes
    model.history = data["history"]
    model.is_fitted = True

    return model

  # --------------------------------------------------------
  # FITTED CHECK
  # --------------------------------------------------------
  def _check_fitted(self):
    if not self.is_fitted:
      raise RuntimeError("Train or load the model first.")
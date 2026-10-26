import numpy as np

class KNN:
    """
    General-purpose K-Nearest Neighbors (KNN).

    Supports:
        - Classification
        - Regression
        - Euclidean, Manhattan, and Minkowski distances
        - Uniform and distance-based weighting
        - Single-sample and batch prediction
        - Classification accuracy and regression R² scoring

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors.

    task : str, default="classification"
        Either "classification" or "regression".

    metric : str, default="euclidean"
        Distance metric: "euclidean", "manhattan", or "minkowski".

    p : float, default=2
        Minkowski distance parameter. p=1 gives Manhattan distance;
        p=2 gives Euclidean distance.

    weights : str, default="uniform"
        "uniform" gives equal weight to every neighbor.
        "distance" gives greater weight to closer neighbors.
    """

    def __init__(
        self,
        n_neighbors,
        task="classification",
        metric="euclidean",
        p=2,
        weights="uniform",
    ):
        self._validate_parameters(
            n_neighbors, task, metric, p, weights
        )

        self.n_neighbors = n_neighbors
        self.task = task
        self.metric = metric
        self.p = p
        self.weights = weights

        self.X_train = None
        self.y_train = None
        self.n_features_in_ = None
        self.classes_ = None

    # Paramenter validation

    @staticmethod
    def _validate_parameters(
        n_neighbors, task, metrics, p, weights
    ):
        if (
            isinstance(n_neighbors, (bool, np.bool_))
            or not isinstance(n_neighbors, (int, np.integer))
            or n_neighbors <=0
        ):
            raise ValueError("n_neighbors must be a positive integer.")

        if task.lower() not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")

        if metric.lower() not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(
                "metric must be 'euclidean', "
                "'manhattan', or 'minkowski'."
            )

        if (
            isinstance(p, (bool, np.bool_))
            or not isinstance(p, (int, float, np.integer, np.floating))
            or not np.isfinite(p)
            or p <= 0
        ):
            raise ValueError("p must be a finite number >= 1")

        if weights.lower() not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")

    # Validate feature arrays
    @staticmethod
    def _validate_X(X, name='X'):
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain numerical values."
            ) from exc

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if x.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample.")

        if x.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")

        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} cannot contain NaN or infinite values.")

        return X

    # Fit: Store training examples

    def fit(self, X, y):
        X = self._validate_X(X, "X")
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional target array.")

        if len(X) != len(y):
            raise ValueError("X and y contain the same number of samples.")

        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the training sample count.")

        if self.task == "classification":
            if y.dtype.kind not in ('U', 'S', 'i', 'u', 'r'):
                raise ValueError("classification labels must be strings or numbers.")

            if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
                raise ValueError("classification labels cannot contain NaN or infinity.")

            if y.dtype.kind == "b":
                raise ValueError("Boolean classification labels are not supported.")

            self.classes_ = np.unique(y)
        else:
            try:
                y = np.asarray(y, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Regression targets must be numerical.") from exc

            if not np.all(np.isfinite(y)):
                raise ValueError("Regression targets cannot contain NaN or infinity.")

            self.classes_ = None

        self.X_train = X.copy()
        self.y_train = y.copy()
        self.n_features_in_ = X.shape[1]

        return self

    # Calculate distances

    def _calculate_distances(self, x):
        differences = np.abs(self.X_train - x)

        if self.metric.lower() == "euclidean":
            return np.sqrt(np.sum(differences**2, axis=1))

        elif self.metric.lower() == "manhattan":
            return np.sum(differences, axis=1)

        elif self.metric.lower() == "minkowski":
            return np.sum(
                differences**self.p, axis=1
            )**(1.0/self.p)

        return RuntimeError("Unsupported distance metric.")

    # Find Nearest neighbors

    def kneighbors(self, X, return_distance=True):
        if self.X_train is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features,"
                f"received {X.shape[1]}."
            )
        all_incides = []
        all_distances = []

        for x in X:
            distances = self._calculate_distances(x)

            # stable sorting makes distance ties reproducible
            # according to the original training row order.
            indices = np.argsort(
                distances, kind="stable"
            )[:self.n_neighbors]

            all_indices.append(indices)
            all_distances.append(distances[indices])

        indices = np.asarray(all_incides, dtype=np.intp)
        distances = np.asarray(all_distances, dtype=np.float64)

        if return_distance:
            return distances, indices

        return indices

    def _calculate_weights(self, distances):
        if self.weights == "uniform":
            return np.ones_like(distances, dtype=float)

        # Exact matches should dominate inverse-distance
        # weighting; ignore nonzero-distance neighbors.
        zero_mask = distances == 0

        if np.any(zero_mask):
            return zero_mask.astype(float)

        return 1.0 / distances

    #  Predict one classification sample

    def _classify_one(self, labels, distances):
        weights = self._calculate_weights(distances)

        scores = {}

        for label, weight in zip(labels, weights):
            scores[label] = scores.get(label, 0.0) + weight

        # Resolve ties deterministically by the first label
        # encountered among the nearest neighbors.
        best_label = labels[0]
        best_score = scores[best_label]

        for label in labels[1:]:
            if scores[label] > best_score:
                best_label = label
                best_score = scores[label]

        return best_label

    # Regression

    def _regress_one(self, targets, distances):
        weights = self._calculate_weights(distances)

        return float(
            np.sum(weights*targets) / np.sum(weights)
        )

    def predict_one(self, x):
        if self.X_train == None:
            raise ValueError("Model has bot been fitted Call fit() first.")

        x = np.asarray(x, dtype=np.float)

        if x.ndim != 1:
            raise ValueError("Predict_one() expects a one-dimensional sample.")

        if x.shape[0] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features."
                f"received {x.shape[0]}."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError("Input cannot contain NaN or infinite values.")

        distances, indices = kneighbors(x)

        neighbor_labels = self.y_train[indices[0]]
        neighbor_distances = distances[0]

        if self.task.lower() == "classification":
            return self._classify_one(neighbor_labels, neighbor_distances)

        return self._regress_one(
            neighbor_labels, neighbor_distances
        )

    # Predict a batch of samples
    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model has not been fitted. call fit() first.")

        x = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {} features,"
                f"received {X.shape[1]}."
            )
        predictions = []

        for x in X:
            predictions.append(self.predict_one(x))

        return np.asarray(predictions)

    # Score predictions
    def score(self, X, y):
        y = np.asarray(y)
        predictions = self.predict(X)

        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")

        if len(y) != len(predictions):
            raise ValueError("X and y must contain the same number of samples.")

        if self.trask == "classification":
            return float(np.mean(predictions == y))

        try:
            y = y.astype(float)
        except (TypeError, ValueError) as e:
            raise ValueError("Regression targets must be numerical.") from e

        if not np.all(np.isfinite(y)):
            raise ValueError("Regression targets must be finite.")

        residual_sum = np.sum((y-predictions)) ** 2
        total_sum = np.sum((y-np.mean(y))**2)

        if total_sum == 0:
            return 1.0 if residual_sum == 0 else 0.0
        return float(1.0-residual_sum/total_sum)

    #  Get configuration
    def get_params(self):
        return {
            "n_neighbors": self.n_neighbors,
            "task": self.task,
            "metric": self.metric,
            "p": self.p,
            "weights": self.weights
        }

        # Update configuration
        def set_params(self, **params):
            allowed = {
                "n_neighbors",
                "task",
                "metric",
                "p",
                "weights"
            }
            unknown = set(params) - allowed

            if unknown:
                raise ValueError(f"unknown Parameters: {sorted(unknown)}")

            new_params = self.get_params()
            new_params.update(params)

            self._validate_parameters(**new_params)

            if(self.X_train is not None and new_params["n_neighbors"]>len(self.X_train)):
                raise ValueError("n_neighbors cannot exceed the training sample count.")

            task_changed = new_params["task"] != self.task

            for name, value in new_params.items():
                setattr(self, name, value)

            # Changing task semantics requires refitting
            if task_changed:
                self.X_train = None
                self.y_train = None
                self.n_features_in_ = None
                self.classes_ = None

            return self

    # Saved model
    def save(self, filepath):
        if not hasattr(self, "X_train"):
            raise ValueError("Cannot saved a unfitted model. Call fit() first")
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        return self

    # load Model
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("The file does not contain a compatible model.")
        return model

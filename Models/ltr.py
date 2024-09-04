import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr

# Load data
ds = tfds.load("mslr_web/10k_fold1", split="train")
ds = ds.map(
    lambda feature_map: {
        "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
        **feature_map,
    }
)
ds = ds.shuffle(buffer_size=1000).padded_batch(batch_size=32)
ds = ds.map(
    lambda feature_map: (
        feature_map,
        tf.where(feature_map["_mask"], feature_map.pop("label"), -1.0),
    )
)

# Create a pointwise model
inputs = {"float_features": tf.keras.Input(shape=(None, 136), dtype=tf.float32)}
norm_inputs = [tf.keras.layers.BatchNormalization()(x) for x in inputs.values()]
x = tf.concat(norm_inputs, axis=-1)
for layer_width in [128, 64, 32]:
    x = tf.keras.layers.Dense(units=layer_width)(x)
    x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)

# Compile and train the pointwise model
pointwise_model = tf.keras.Model(inputs=inputs, outputs=scores)
pointwise_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tfr.keras.losses.SoftmaxLoss(),
    metrics=tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"),
)
pointwise_model.fit(ds, epochs=3)


# Create a pairwise model
inputs = {"float_features": tf.keras.Input(shape=(None, 136), dtype=tf.float32)}
norm_inputs = [tf.keras.layers.BatchNormalization()(x) for x in inputs.values()]
x = tf.concat(norm_inputs, axis=-1)
for layer_width in [128, 64, 32]:
    x = tf.keras.layers.Dense(units=layer_width)(x)
    x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)

# Compile and train the pairwise model
pairwise_model = tf.keras.Model(inputs=inputs, outputs=scores)
pairwise_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tfr.keras.losses.get("pairwise_logistic_loss"),
    metrics=tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"),
)
pairwise_model.fit(ds, epochs=3)


# Create a listwise model
inputs = {"float_features": tf.keras.Input(shape=(None, 136), dtype=tf.float32)}
norm_inputs = [tf.keras.layers.BatchNormalization()(x) for x in inputs.values()]
x = tf.concat(norm_inputs, axis=-1)
for layer_width in [128, 64, 32]:
    x = tf.keras.layers.Dense(units=layer_width)(x)
    x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)

# Compile and train the listwise model
listwise_model = tf.keras.Model(inputs=inputs, outputs=scores)
listwise_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tfr.keras.losses.get("approx_ndcg_loss"),
    metrics=tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"),
)
listwise_model.fit(ds, epochs=3)

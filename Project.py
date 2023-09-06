import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


mydb = mysql.connector.connect(
  host="localhost",
  port = 3306,
  user="root",
  password="encrypted1122",
  database="classicmodels"
)

query = "SELECT * FROM orders"
cursor = mydb.cursor()
cursor.execute(query)
result = cursor.fetchall()
orders_df = pd.DataFrame(result, columns=cursor.column_names)
print(cursor.column_names)
orders_df['orderDate'] = pd.to_datetime(orders_df['orderDate'])
orders_df['requiredDate'] = pd.to_datetime(orders_df['requiredDate'])
orders_df['shippedDate'] = pd.to_datetime(orders_df['shippedDate'])

# Create a new column for shipping delay
orders_df['shippingDelay'] = (orders_df['requiredDate'] - orders_df['shippedDate']).dt.days
orders_df.drop(columns=['comments', 'orderNumber', 'customerNumber'], axis =1, inplace=True)
orders_df.dropna(inplace=True)
le = LabelEncoder()
orders_df['status'] = le.fit_transform(orders_df['status'])
print(orders_df.head())

# Time-based trend for orders
orders_df.set_index('orderDate').resample('M').size().plot()
plt.title('Number of Orders Over Time')
plt.xlabel('Order Date')
plt.ylabel('Number of Orders')
plt.show()

# Distribution of shipping delays
sns.histplot(orders_df['shippingDelay'].dropna(), kde=True)
plt.title('Distribution of Shipping Delays')
plt.xlabel('Days Delayed')
plt.ylabel('Frequency')
plt.show()

# Status vs Shipping Delay
sns.boxplot(x='status', y='shippingDelay', data=orders_df)
plt.title('Shipping Delay by Order Status')
plt.xticks(rotation=45)
plt.show()

# 1. Calculate days between dates
orders_df['days_to_required'] = (orders_df['requiredDate'] - orders_df['orderDate']).dt.days
orders_df['days_to_shipped'] = (orders_df['shippedDate'] - orders_df['orderDate']).dt.days
orders_df['days_required_to_shipped'] = (orders_df['shippedDate'] - orders_df['requiredDate']).dt.days

# 2. Extract year, month, day, weekday
orders_df['orderDate_year'] = orders_df['orderDate'].dt.year
orders_df['orderDate_month'] = orders_df['orderDate'].dt.month
orders_df['orderDate_day'] = orders_df['orderDate'].dt.day
orders_df['orderDate_weekday'] = orders_df['orderDate'].dt.weekday

# 3. Is it a weekend?
orders_df['order_weekend'] = orders_df['orderDate_weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Display the updated DataFrame
orders_df.drop(columns=['orderDate', 'requiredDate', 'shippedDate'], axis=1, inplace=True)
print(orders_df.columns)
print(orders_df.info())

# Feature and Target variables
X = orders_df[['shippingDelay', 'days_to_required', 'days_to_shipped',
               'days_required_to_shipped', 'orderDate_year', 'orderDate_month',
               'orderDate_day', 'orderDate_weekday', 'order_weekend']]
y = orders_df['status']

# Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Combine training data back together
#train_data = X_train.copy()
#train_data['status'] = y_train

# Separate majority and minority classes
#majority = train_data[train_data.status == 3]
#minorities = [train_data[train_data.status == i] for i in [0, 1, 2]]

#from sklearn.utils import resample # Upsample minority classes
#upsampled_minorities = [resample(minority, replace=True, n_samples=len(majority), random_state=42) for minority in minorities]

# Combine majority class with upsampled minority classes
#train_upsampled = pd.concat([majority] + upsampled_minorities)

# Split back into X and y
#y_train_upsampled = train_upsampled['status']
#X_train_upsampled = train_upsampled.drop('status', axis=1)

# After the upsampling process, update the proportions in the training set
#print("Train set after upsampling:")
#print(y_train_upsampled.value_counts(normalize=True))

# Your test set remains the same
#print("Test set:")
#print(y_test.value_counts(normalize=True))
y_train_binary = y_train.apply(lambda x: 1 if x == 3 else 0)
y_test_binary = y_test.apply(lambda x: 1 if x == 3 else 0)
#print(y_test_binary.value_counts())
#print(y_train_binary.value_counts())


# Create the model with class weight
#model = RandomForestClassifier()
# Train the model
#model.fit(X_train, y_train_binary)

# Prediction and Evaluation
#y_pred = model.predict(X_test)
#print(classification_report(y_test_binary, y_pred))

# Create the model
#model2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the model
#model2.fit(X_train, y_train_binary)

# Prediction and Evaluation
#y_pred2 = model2.predict(X_test)

#print(classification_report(y_test_binary, y_pred2))

import tensorflow as tf
import numpy as np

# Isolate Minority Class (Only 7 samples)
X_train_minority = X_train[y_train_binary == 0]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_minority_scaled = scaler.fit_transform(X_train_minority)

# Number of features in X_train
n_features = X_train_minority_scaled.shape[1]


# Step 3: Create GAN Model
# Build Generator
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(n_features, activation='sigmoid')
    ])
    return model


# Build Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


generator = build_generator()
discriminator = build_discriminator()

# Compile Discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')


# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


gan = build_gan(generator, discriminator)

# Step 4: Train the GAN
batch_size = 128
labels_real = np.ones((batch_size, 1))
labels_fake = np.zeros((batch_size, 1))

for epoch in range(10000):
    idx = np.random.randint(0, X_train_minority_scaled.shape[0], batch_size)
    real_samples = X_train_minority_scaled[idx]

    noise = np.random.normal(0, 1, (batch_size, n_features))
    generated_samples = generator.predict(noise)

    # Discriminator training
    d_loss_real = discriminator.train_on_batch(real_samples, labels_real)
    d_loss_fake = discriminator.train_on_batch(generated_samples, labels_fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generator training
    noise = np.random.normal(0, 1, (batch_size, n_features))
    g_loss = gan.train_on_batch(noise, labels_real)

    if epoch % 1000 == 0:
        print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
# ...

# Generate synthetic samples
n_synthetic_samples = 220  # Number of synthetic samples to generate
noise = np.random.normal(0, 1, (n_synthetic_samples, n_features))
generated_samples = generator.predict(noise)

# Revert feature scaling on generated samples
generated_samples = scaler.inverse_transform(generated_samples)

# Combine with original dataset
X_train_balanced = np.vstack((X_train, generated_samples))
y_train_balanced = np.hstack((y_train_binary, np.zeros((n_synthetic_samples,))))








#from sklearn.model_selection import LeaveOneOut
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#import numpy as np

#loo = LeaveOneOut()

#accuracies = []

#for train_index, test_index in loo.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]

#    model = RandomForestClassifier()
#    model.fit(X_train, y_train)

#    y_pred = model.predict(X_test)
#    accuracies.append(accuracy_score(y_test, y_pred))

#average_accuracy = np.mean(accuracies)
#print(f'Average accuracy: {average_accuracy}')


#from sklearn.model_selection import KFold
#kf = KFold(n_splits=3)  # n_splits is the number of folds

#accuracies = []

#for train_index, test_index in kf.split(X):
 #   X_train, X_test = X[train_index], X[test_index]
 #   y_train, y_test = y[train_index], y[test_index]

 #   model = RandomForestClassifier()
 #   model.fit(X_train, y_train)

#    y_pred = model.predict(X_test)
#    accuracies.append(accuracy_score(y_test, y_pred))

#average_accuracy = np.mean(accuracies)
#print(f'Average accuracy: {average_accuracy}')

# ----------------------------------------------
# Perk Assignment Script (from Part 2.2 onwards)
# Make sure to load 'session_level_based_table_cleaned.csv' as df_cleaned
# ----------------------------------------------

df_cleaned.info()

df_cleaned.describe(include="all")

# Free Hotel Meal
# Filter only sessions where a hotel was booked
df_hotel_bookings = df_cleaned[df_cleaned["hotel_booked"] == True].copy()

# Remove sessions without any hotel nights
df_hotel_bookings = df_hotel_bookings[df_hotel_bookings["calc_nights"] > 3]

# Aggregate by user
df_free_meal_score = df_hotel_bookings.groupby("user_id").agg(
    num_hotel_bookings=("session_id", "nunique"),
    avg_hotel_stay_duration=("calc_nights", "mean"),
    avg_hotel_price=("hotel_per_room_usd", "mean")
).reset_index()

# Normalize values for scoring (0 to 1)
df_free_meal_score["score_num_bookings"] = df_free_meal_score["num_hotel_bookings"] / df_free_meal_score["num_hotel_bookings"].max()
df_free_meal_score["score_stay_duration"] = df_free_meal_score["avg_hotel_stay_duration"] / df_free_meal_score["avg_hotel_stay_duration"].max()
df_free_meal_score["score_price"] = df_free_meal_score["avg_hotel_price"] / df_free_meal_score["avg_hotel_price"].max()

# Weighted perk score
df_free_meal_score["perk_score_free_meal"] = (
    0.5 * df_free_meal_score["score_num_bookings"] +
    0.3 * df_free_meal_score["score_stay_duration"] +
    0.2 * df_free_meal_score["score_price"]
)

# Dynamic eligibility for Free Hotel Meal
threshold_meal = df_free_meal_score["perk_score_free_meal"].quantile(0.85)
df_free_meal_score["eligible_free_meal"] = df_free_meal_score["perk_score_free_meal"] >= threshold_meal

# ---

# Free Checked Bag
# Filter only sessions where a flight was booked
df_flight_bag_sessions = df_cleaned[df_cleaned["flight_booked"] == True].copy()

# Remove sessions without valid checked bag data
df_flight_bag_sessions = df_flight_bag_sessions[df_flight_bag_sessions["checked_bags"] >= 1]

# Aggregate metrics per user
df_perk_score_checked_bag = df_flight_bag_sessions.groupby("user_id").agg(
    num_flight_bookings=("session_id", "nunique"),
    avg_checked_bags=("checked_bags", "mean"),
    avg_base_fare=("base_fare_usd", "mean")
).reset_index()

# Normalize values to scale between 0 and 1
df_perk_score_checked_bag["score_flights"] = df_perk_score_checked_bag["num_flight_bookings"] / df_perk_score_checked_bag["num_flight_bookings"].max()
df_perk_score_checked_bag["score_bags"] = df_perk_score_checked_bag["avg_checked_bags"] / df_perk_score_checked_bag["avg_checked_bags"].max()
df_perk_score_checked_bag["score_fare"] = df_perk_score_checked_bag["avg_base_fare"] / df_perk_score_checked_bag["avg_base_fare"].max()

# Compute weighted score for the perk
df_perk_score_checked_bag["perk_score_checked_bag"] = (
    0.6 * df_perk_score_checked_bag["score_flights"] +
    0.3 * df_perk_score_checked_bag["score_bags"] +
    0.1 * df_perk_score_checked_bag["score_fare"]
)

# Dynamic eligibility for free checked bag
threshold_bag = df_perk_score_checked_bag["perk_score_checked_bag"].quantile(0.85)
df_perk_score_checked_bag["eligible_free_checked_bag"] = df_perk_score_checked_bag["perk_score_checked_bag"] >= threshold_bag

# ---

# No Cancellation Fees
# Filter only sessions where a booking was made
df_bookings = df_cleaned[df_cleaned["booking"] == True].copy()

# Calculate the days between booking session and actual trip (earliest of check-in or departure)
df_bookings["trip_start_time"] = df_bookings[["check_in_time", "departure_time"]].min(axis=1)
df_bookings["days_until_trip"] = (df_bookings["trip_start_time"] - df_bookings["session_start"]).dt.days
df_bookings["days_until_trip"] = df_bookings["days_until_trip"].clip(lower=0)

# Group by user and compute metrics
df_perk_score_no_cancel = df_bookings.groupby("user_id").agg(
    total_bookings=("session_id", "nunique"),
    num_cancellations=("cancellation", "sum"),
    avg_membership_duration=("membership_duration_days", "mean"),
    avg_days_until_trip=("days_until_trip", "mean")
).reset_index()

# Calculate cancellation rate
df_perk_score_no_cancel["cancellation_rate"] = (
    df_perk_score_no_cancel["num_cancellations"] / df_perk_score_no_cancel["total_bookings"]
)

# Normalize all metrics
df_perk_score_no_cancel["score_booking_volume"] = (
    df_perk_score_no_cancel["total_bookings"] / df_perk_score_no_cancel["total_bookings"].max()
)
df_perk_score_no_cancel["score_loyalty"] = (
    df_perk_score_no_cancel["avg_membership_duration"] / df_perk_score_no_cancel["avg_membership_duration"].max()
)
df_perk_score_no_cancel["score_planning_ahead"] = (
    df_perk_score_no_cancel["avg_days_until_trip"] / df_perk_score_no_cancel["avg_days_until_trip"].max()
)

# Inverse cancellation rate to reward low cancel behavior
df_perk_score_no_cancel["score_cancellation_behavior"] = 1 - df_perk_score_no_cancel["cancellation_rate"]
df_perk_score_no_cancel["score_cancellation_behavior"] = df_perk_score_no_cancel["score_cancellation_behavior"].clip(lower=0)

# Final score including the new time-based metric
df_perk_score_no_cancel["perk_score_no_cancellation_fees"] = (
    0.3 * df_perk_score_no_cancel["score_booking_volume"] +
    0.3 * df_perk_score_no_cancel["score_cancellation_behavior"] +
    0.2 * df_perk_score_no_cancel["score_loyalty"] +
    0.2 * df_perk_score_no_cancel["score_planning_ahead"]
)

# Determine dynamic threshold (top 25%)
threshold_cancel = df_perk_score_no_cancel["perk_score_no_cancellation_fees"].quantile(0.85)
df_perk_score_no_cancel["eligible_no_cancellation_fees"] = (
    df_perk_score_no_cancel["perk_score_no_cancellation_fees"] >= threshold_cancel
)

# ---

# Exclusive Discounts
# Filter only sessions with at least one booking
df_booked_sessions = df_cleaned[df_cleaned["booking"] == True].copy()

# Further restrict to users with more than 3 bookings
booking_counts = df_booked_sessions.groupby("user_id")["session_id"].nunique().reset_index(name="total_bookings")
eligible_users = booking_counts[booking_counts["total_bookings"] > 3]["user_id"]

# Filter users who never used any kind of discount
df_no_discount_users = df_cleaned[
    (df_cleaned["user_id"].isin(eligible_users)) &
    (df_cleaned["discount"] == "no") &
    (df_cleaned["flight_discount"] == False) &
    (df_cleaned["hotel_discount"] == False)
]

# Group by user to calculate final features
df_perk_score_exclusive_discounts = df_no_discount_users.groupby("user_id").agg(
    total_sessions=("session_id", "count"),
    avg_membership_duration=("membership_duration_days", "mean"),
    total_spend=("base_fare_usd", "sum")
).reset_index()

# Normalize each metric
df_perk_score_exclusive_discounts["score_sessions"] = (
    df_perk_score_exclusive_discounts["total_sessions"] / df_perk_score_exclusive_discounts["total_sessions"].max()
)
df_perk_score_exclusive_discounts["score_loyalty"] = (
    df_perk_score_exclusive_discounts["avg_membership_duration"] / df_perk_score_exclusive_discounts["avg_membership_duration"].max()
)
df_perk_score_exclusive_discounts["score_spend"] = (
    df_perk_score_exclusive_discounts["total_spend"] / df_perk_score_exclusive_discounts["total_spend"].max()
)

# Weighted final score
df_perk_score_exclusive_discounts["perk_score_exclusive_discounts"] = (
    0.4 * df_perk_score_exclusive_discounts["score_sessions"] +
    0.3 * df_perk_score_exclusive_discounts["score_loyalty"] +
    0.3 * df_perk_score_exclusive_discounts["score_spend"]
)

# Dynamic threshold (e.g. top 25%)
threshold_discounts = df_perk_score_exclusive_discounts["perk_score_exclusive_discounts"].quantile(0.85)
df_perk_score_exclusive_discounts["eligible_exclusive_discounts"] = (
    df_perk_score_exclusive_discounts["perk_score_exclusive_discounts"] >= threshold_discounts
)

# ---

# Hotel + Flight Combo (Free Night)
# Filter only sessions where both hotel and flight were booked
df_combo_trips = df_cleaned[
    (df_cleaned["hotel_booked"] == True) &
    (df_cleaned["flight_booked"] == True) &
    (df_cleaned["calc_nights"] >= 2)
].copy()

# Aggregate per user
df_perk_score_hotel_flight = df_combo_trips.groupby("user_id").agg(
    num_combined_trips=("session_id", "nunique"),
    avg_price_flight=("base_fare_usd", "mean"),
    avg_price_hotel=("hotel_per_room_usd", "mean")
).reset_index()

# Normalize values
df_perk_score_hotel_flight["score_combined"] = df_perk_score_hotel_flight["num_combined_trips"] / df_perk_score_hotel_flight["num_combined_trips"].max()
df_perk_score_hotel_flight["score_flight_price"] = df_perk_score_hotel_flight["avg_price_flight"] / df_perk_score_hotel_flight["avg_price_flight"].max()
df_perk_score_hotel_flight["score_hotel_price"] = df_perk_score_hotel_flight["avg_price_hotel"] / df_perk_score_hotel_flight["avg_price_hotel"].max()

# Compute the final weighted score
df_perk_score_hotel_flight["perk_score_hotel_flight_combo"] = (
    0.5 * df_perk_score_hotel_flight["score_combined"] +
    0.25 * df_perk_score_hotel_flight["score_flight_price"] +
    0.25 * df_perk_score_hotel_flight["score_hotel_price"]
)

# Dynamic eligibility for one night free with hotel
threshold_combo = df_perk_score_hotel_flight["perk_score_hotel_flight_combo"].quantile(0.85)
df_perk_score_hotel_flight["eligible_free_night_combo"] = df_perk_score_hotel_flight["perk_score_hotel_flight_combo"] >= threshold_combo

# ---

# Welcome Drink for No-Booker Users
# Create 'booking' column from flight or hotel bookings
df_cleaned["booking"] = df_cleaned["hotel_booked"] | df_cleaned["flight_booked"]

# Identify true no-booker users (no bookings, no cancellations)
df_no_bookers = df_cleaned.groupby("user_id").agg(
    has_booking=("booking", "any"),
    has_cancellation=("cancellation", "any")
).reset_index()

# Only keep users with no booking and no cancellation activity
df_no_bookers = df_no_bookers[
    (df_no_bookers["has_booking"] == False) &
    (df_no_bookers["has_cancellation"] == False)
]

# Collect session data for these users
df_no_booker_sessions = df_cleaned[df_cleaned["user_id"].isin(df_no_bookers["user_id"])]

# Aggregate engagement features per user
df_perk_score_welcome_drink = df_no_booker_sessions.groupby("user_id").agg(
    total_sessions=("session_id", "count"),
    total_page_clicks=("page_clicks", "sum"),
    avg_session_duration=("session_duration_minutes", "mean"),
    membership_days=("membership_duration_days", "mean")
).reset_index()

# Normalize all values
df_perk_score_welcome_drink["score_sessions"] = df_perk_score_welcome_drink["total_sessions"] / df_perk_score_welcome_drink["total_sessions"].max()
df_perk_score_welcome_drink["score_clicks"] = df_perk_score_welcome_drink["total_page_clicks"] / df_perk_score_welcome_drink["total_page_clicks"].max()
df_perk_score_welcome_drink["score_duration"] = df_perk_score_welcome_drink["avg_session_duration"] / df_perk_score_welcome_drink["avg_session_duration"].max()
df_perk_score_welcome_drink["score_membership"] = df_perk_score_welcome_drink["membership_days"] / df_perk_score_welcome_drink["membership_days"].max()

# Calculate weighted score
df_perk_score_welcome_drink["perk_score_welcome_drink"] = (
    0.3 * df_perk_score_welcome_drink["score_sessions"] +
    0.3 * df_perk_score_welcome_drink["score_clicks"] +
    0.2 * df_perk_score_welcome_drink["score_duration"] +
    0.2 * df_perk_score_welcome_drink["score_membership"]
)

# Define eligibility flag
df_perk_score_welcome_drink["eligible_free_welcome_drink"] = df_perk_score_welcome_drink["perk_score_welcome_drink"] >= 0.3

# ---

# Combine all perk scores into a single user profile DataFrame
# Start with Free Hotel Meal perks as base
df_user_profile = df_free_meal_score[["user_id", "perk_score_free_meal", "eligible_free_meal"]].copy()

# Merge Free Checked Bag scores
df_user_profile = df_user_profile.merge(
    df_perk_score_checked_bag[["user_id", "perk_score_checked_bag", "eligible_free_checked_bag"]],
    on="user_id", how="outer"
)

# Merge No Cancellation Fees scores
df_user_profile = df_user_profile.merge(
    df_perk_score_no_cancel[["user_id", "perk_score_no_cancellation_fees", "eligible_no_cancellation_fees"]],
    on="user_id", how="outer"
)

# Merge Exclusive Discounts scores
df_user_profile = df_user_profile.merge(
    df_perk_score_exclusive_discounts[["user_id", "perk_score_exclusive_discounts", "eligible_exclusive_discounts"]],
    on="user_id", how="outer"
)

# Merge Hotel + Flight Combo (Free Night) scores
df_user_profile = df_user_profile.merge(
    df_perk_score_hotel_flight[["user_id", "perk_score_hotel_flight_combo", "eligible_free_night_combo"]],
    on="user_id", how="outer"
)

# Merge Welcome Drink for no-booker users
df_user_profile = df_user_profile.merge(
    df_perk_score_welcome_drink[["user_id", "perk_score_welcome_drink", "eligible_free_welcome_drink"]],
    on="user_id", how="outer"
)

# Define all other perk score columns (except welcome drink)
non_welcome_scores = [
    "perk_score_free_meal",
    "perk_score_checked_bag",
    "perk_score_no_cancellation_fees",
    "perk_score_exclusive_discounts",
    "perk_score_hotel_flight_combo"
]

# Recalculate booking status if not already done
df_cleaned["booking"] = df_cleaned["hotel_booked"] | df_cleaned["flight_booked"]

# Identify true no-booker users from raw session behavior
df_no_bookers_final = df_cleaned.groupby("user_id").agg(
    has_booking=("booking", "any"),
    has_cancellation=("cancellation", "any")
).reset_index()

# Extract only users with no bookings and no cancellations
no_booker_ids = df_no_bookers_final[
    (df_no_bookers_final["has_booking"] == False) &
    (df_no_bookers_final["has_cancellation"] == False)
]["user_id"]

# Flag them in the final user profile
df_user_profile["is_no_booker"] = df_user_profile["user_id"].isin(no_booker_ids)

# Basic shape and column list
print("Shape:", df_user_profile.shape)
print("Columns:", df_user_profile.columns.tolist())

# Identify perk score and eligibility columns
perk_score_cols = [col for col in df_user_profile.columns if col.startswith("perk_score")]
perk_eligibility_cols = [col for col in df_user_profile.columns if col.startswith("eligible_")]

# Descriptive statistics for all perk scores
print("\nPerk Score Summary:")
print(df_user_profile[perk_score_cols].describe().T)

# Count how many users are eligible per perk
print("\nEligibility Distribution:")
print(df_user_profile[perk_eligibility_cols].sum().sort_values(ascending=False))

# Count how many users are marked as no bookers
print("\nNo Booker Stats:")
print("No Bookers:", df_user_profile["is_no_booker"].sum())
print("Total Users:", df_user_profile.shape[0])

df_user_profile.head()

df_user_profile.info()

# df_user_profile.to_csv("/content/drive/MyDrive/user_level_based_table", index=False)

# Reproducibility
import random

np.random.seed(42)
random.seed(42)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import itertools

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Evaluation
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Select relevant demographic features from session-level data
personal_cols = ["user_id", "age", "age_group", "family_status"]
df_personal_info = df_cleaned[personal_cols].drop_duplicates(subset="user_id")

# Merge perk score table with personal info using user_id
df_eda = pd.merge(df_user_profile, df_personal_info, on="user_id", how="left")

df_user_profile.describe()

df_user_profile.describe(include="object")

# Define the score columns
perk_score_columns = [
    "perk_score_free_meal",
    "perk_score_checked_bag",
    "perk_score_no_cancellation_fees",
    "perk_score_exclusive_discounts",
    "perk_score_hotel_flight_combo",
    "perk_score_welcome_drink"
]

# Plot histograms for each perk score
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(perk_score_columns):
    sns.histplot(df_eda[col].dropna(), kde=True, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel("Score")
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

# Plot boxplots for each perk score grouped by age group
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(perk_score_columns):
    sns.boxplot(
        data=df_eda,
        x="age_group",
        y=col,
        ax=axes[i],
        order=sorted(df_eda["age_group"].dropna().unique())
    )
    axes[i].set_title(f"{col} by Age Group")
    axes[i].set_xlabel("Age Group")
    axes[i].set_ylabel("Score")
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Compute and plot correlation matrix of perk scores
perk_scores_only = df_eda[perk_score_columns]
correlation_matrix = perk_scores_only.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Matrix of Perk Scores")
plt.tight_layout()
plt.show()

# Define the perk score columns
score_cols = [
    "perk_score_free_meal",
    "perk_score_checked_bag",
    "perk_score_no_cancellation_fees",
    "perk_score_exclusive_discounts",
    "perk_score_hotel_flight_combo",
    "perk_score_welcome_drink"
]

# Define eligibility columns
eligibility_cols = [
    "eligible_free_meal",
    "eligible_free_checked_bag",
    "eligible_no_cancellation_fees",
    "eligible_exclusive_discounts",
    "eligible_free_night_combo",
    "eligible_free_welcome_drink"
]

# Create working copy for clustering
df_cluster_elig = df_user_profile.copy()

# Fill missing perk scores
df_cluster_elig[score_cols] = df_cluster_elig[score_cols].fillna(0)

# Convert eligibility to binary + fill missing
df_cluster_elig[eligibility_cols] = (
    df_cluster_elig[eligibility_cols].astype(bool).astype(int).fillna(0)
)

# Extract full feature sets
df_cluster_scores_only = df_cluster_elig[score_cols]
df_cluster_with_elig = df_cluster_elig[score_cols + eligibility_cols]

# Standardize features
scaler = StandardScaler()
X_scores_scaled = scaler.fit_transform(df_cluster_scores_only)
X_with_elig_scaled = scaler.fit_transform(df_cluster_with_elig)

# --- Global Split for all cluster methods ---
mask_non_nobooker = df_cluster_elig["is_no_booker"] == False
mask_nobooker = df_cluster_elig["is_no_booker"] == True

df_non_nobooker = df_cluster_elig[mask_non_nobooker].copy()
df_nobooker = df_cluster_elig[mask_nobooker].copy()

X_scores_non_nobooker = X_scores_scaled[mask_non_nobooker]
X_with_elig_non_nobooker = X_with_elig_scaled[mask_non_nobooker]

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scores_non_nobooker)

df_non_nobooker["cluster_scores"] = labels
df_nobooker["cluster_scores"] = 5
df_clustered_scores = pd.concat([df_non_nobooker, df_nobooker], ignore_index=True)

print("KMeans – Scores only – No-Booker separated")
print("Silhouette:", round(silhouette_score(X_scores_non_nobooker, labels), 4))
print("DBI:", round(davies_bouldin_score(X_scores_non_nobooker, labels), 4))
print(df_clustered_scores["cluster_scores"].value_counts().sort_index())

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_with_elig_non_nobooker)

df_non_nobooker["cluster_scores_elig"] = labels
df_nobooker["cluster_scores_elig"] = 5
df_clustered_scores_elig = pd.concat([df_non_nobooker, df_nobooker], ignore_index=True)

print("KMeans – Scores + Eligibility – No-Booker separated")
print("Silhouette:", round(silhouette_score(X_with_elig_non_nobooker, labels), 4))
print("DBI:", round(davies_bouldin_score(X_with_elig_non_nobooker, labels), 4))
print(df_clustered_scores_elig["cluster_scores_elig"].value_counts().sort_index())

kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(X_scores_scaled)

df_cluster_elig["cluster_scores_k6"] = labels

print("KMeans – Scores only – No-Booker included")
print("Silhouette:", round(silhouette_score(X_scores_scaled, labels), 4))
print("DBI:", round(davies_bouldin_score(X_scores_scaled, labels), 4))
print(df_cluster_elig["cluster_scores_k6"].value_counts().sort_index())

kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(X_with_elig_scaled)

df_cluster_elig["cluster_scores_elig_k6"] = labels

print("KMeans – Scores + Eligibility – No-Booker included")
print("Silhouette:", round(silhouette_score(X_with_elig_scaled, labels), 4))
print("DBI:", round(davies_bouldin_score(X_with_elig_scaled, labels), 4))
print(df_cluster_elig["cluster_scores_elig_k6"].value_counts().sort_index())

print("DBSCAN – Scores only (no-bookers separated)\n")

results = []
for eps, min_samples in itertools.product(np.arange(0.1, 1.5, 0.1), range(1, 11)):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scores_non_nobooker)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 5:
        mask_valid = labels != -1
        if mask_valid.sum() == 0:
            continue

        sil = silhouette_score(X_scores_non_nobooker[mask_valid], labels[mask_valid])
        dbi = davies_bouldin_score(X_scores_non_nobooker[mask_valid], labels[mask_valid])
        results.append((sil, dbi, eps, min_samples))

top_5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
for i, (sil, dbi, eps, ms) in enumerate(top_5, 1):
    print(f"{i}. eps={eps:.2f}, min_samples={ms} → Silhouette: {sil:.4f}, DBI: {dbi:.4f}")

print("\nDBSCAN – Scores + Eligibility (no-bookers separated)\n")

results = []
for eps, min_samples in itertools.product(np.arange(0.1, 2.6, 0.1), range(1, 11)):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_with_elig_non_nobooker)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 5:
        mask_valid = labels != -1
        if mask_valid.sum() == 0:
            continue

        sil = silhouette_score(X_with_elig_non_nobooker[mask_valid], labels[mask_valid])
        dbi = davies_bouldin_score(X_with_elig_non_nobooker[mask_valid], labels[mask_valid])
        results.append((sil, dbi, eps, min_samples))

top_5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
for i, (sil, dbi, eps, ms) in enumerate(top_5, 1):
    print(f"{i}. eps={eps:.2f}, min_samples={ms} → Silhouette: {sil:.4f}, DBI: {dbi:.4f}")

print("\nDBSCAN – Scores only (no-bookers included)\n")

results = []
for eps, min_samples in itertools.product(np.arange(0.1, 1.5, 0.1), range(1, 11)):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scores_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 6:
        mask_valid = labels != -1
        if mask_valid.sum() == 0:
            continue

        sil = silhouette_score(X_scores_scaled[mask_valid], labels[mask_valid])
        dbi = davies_bouldin_score(X_scores_scaled[mask_valid], labels[mask_valid])
        results.append((sil, dbi, eps, min_samples))

top_5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
for i, (sil, dbi, eps, ms) in enumerate(top_5, 1):
    print(f"{i}. eps={eps:.2f}, min_samples={ms} → Silhouette: {sil:.4f}, DBI: {dbi:.4f}")

print("\nDBSCAN – Scores + Eligibility (no-bookers included)\n")

results = []
for eps, min_samples in itertools.product(np.arange(0.1, 2.6, 0.1), range(1, 11)):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_with_elig_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 6:
        mask_valid = labels != -1
        if mask_valid.sum() == 0:
            continue

        sil = silhouette_score(X_with_elig_scaled[mask_valid], labels[mask_valid])
        dbi = davies_bouldin_score(X_with_elig_scaled[mask_valid], labels[mask_valid])
        results.append((sil, dbi, eps, min_samples))

top_5 = sorted(results, key=lambda x: x[0], reverse=True)[:5]
for i, (sil, dbi, eps, ms) in enumerate(top_5, 1):
    print(f"{i}. eps={eps:.2f}, min_samples={ms} → Silhouette: {sil:.4f}, DBI: {dbi:.4f}")

# Clustering only non-no-booker users
agglo_scores = AgglomerativeClustering(n_clusters=5, linkage="ward")
labels_scores_agglo = agglo_scores.fit_predict(X_scores_scaled[df_non_nobooker.index])

# Cluster
df_non_nobooker["cluster_scores_agglo"] = labels_scores_agglo
df_nobooker["cluster_scores_agglo"] = 5  # no-bookers = Cluster 5

# Combine into one DataFrame
df_clustered_scores_agglo = pd.concat([df_non_nobooker, df_nobooker], ignore_index=True)

# Evaluate
sil_score_agglo = silhouette_score(X_scores_scaled[df_non_nobooker.index], labels_scores_agglo)
dbi_score_agglo = davies_bouldin_score(X_scores_scaled[df_non_nobooker.index], labels_scores_agglo)

# Output
print("Agglomerative Clustering – Perk Scores (no-booker separat)")
print("Silhouette Score:", round(sil_score_agglo, 4))
print("Davies-Bouldin Index:", round(dbi_score_agglo, 4))
print("\nCluster Distribution (cluster_scores_agglo):")
print(df_clustered_scores_agglo["cluster_scores_agglo"].value_counts().sort_index())

# Clustering only non-no-booker users with eligibility
agglo_scores_elig = AgglomerativeClustering(n_clusters=5, linkage="ward")
labels_scores_elig_agglo = agglo_scores_elig.fit_predict(X_with_elig_scaled[df_non_nobooker.index])

# Cluster
df_non_nobooker["cluster_scores_elig_agglo"] = labels_scores_elig_agglo
df_nobooker["cluster_scores_elig_agglo"] = 5  # no-bookers = Cluster 5

# Combine
df_clustered_scores_elig_agglo = pd.concat([df_non_nobooker, df_nobooker], ignore_index=True)

# Evaluate
sil_score_elig_agglo = silhouette_score(X_with_elig_scaled[df_non_nobooker.index], labels_scores_elig_agglo)
dbi_score_elig_agglo = davies_bouldin_score(X_with_elig_scaled[df_non_nobooker.index], labels_scores_elig_agglo)

# Output
print("Agglomerative Clustering – Scores + Eligibility (no-booker separat)")
print("Silhouette Score:", round(sil_score_elig_agglo, 4))
print("Davies-Bouldin Index:", round(dbi_score_elig_agglo, 4))
print("\nCluster Distribution (cluster_scores_elig_agglo):")
print(df_clustered_scores_elig_agglo["cluster_scores_elig_agglo"].value_counts().sort_index())

# Clustering on whole df with eligibility (incl. no-booker)
agglo_all_scores = AgglomerativeClustering(n_clusters=6, linkage="ward")
labels_all_scores = agglo_all_scores.fit_predict(X_scores_scaled)

# Cluster
df_clustered_scores_agglo_all = df_cluster_elig.copy()
df_clustered_scores_agglo_all["cluster_scores_agglo"] = labels_all_scores

# Evaluation
sil_score_all = silhouette_score(X_scores_scaled, labels_all_scores)
dbi_score_all = davies_bouldin_score(X_scores_scaled, labels_all_scores)

# Output
print("Agglomerative Clustering – Perk Scores (no-booker included, k=6)")
print("Silhouette Score:", round(sil_score_all, 4))
print("Davies-Bouldin Index:", round(dbi_score_all, 4))
print("\nCluster Distribution (cluster_scores_agglo):")
print(df_clustered_scores_agglo_all["cluster_scores_agglo"].value_counts().sort_index())

# Clustering on whole df with eligibility
agglo_all_scores_elig = AgglomerativeClustering(n_clusters=6, linkage="ward")
labels_all_scores_elig = agglo_all_scores_elig.fit_predict(X_with_elig_scaled)

# Cluster
df_clustered_scores_elig_agglo_all = df_cluster_elig.copy()
df_clustered_scores_elig_agglo_all["cluster_scores_elig_agglo"] = labels_all_scores_elig

# Evaluation
sil_score_elig_all = silhouette_score(X_with_elig_scaled, labels_all_scores_elig)
dbi_score_elig_all = davies_bouldin_score(X_with_elig_scaled, labels_all_scores_elig)

# Output
print("Agglomerative Clustering – Scores + Eligibility (no-booker included, k=6)")
print("Silhouette Score:", round(sil_score_elig_all, 4))
print("Davies-Bouldin Index:", round(dbi_score_elig_all, 4))
print("\nCluster Distribution (cluster_scores_elig_agglo):")
print(df_clustered_scores_elig_agglo_all["cluster_scores_elig_agglo"].value_counts().sort_index())

# Collecting results
cluster_results = {
    "KMeans\nScores\n(separated)": [908, 1019, 1027, 1074, 1514, 456],
    "KMeans\nScores+Elig\n(separated)": [801, 841, 1125, 1299, 1476, 456],
    "KMeans\nScores\n(included)": [1832, 748, 598, 456, 1766, 598],
    "KMeans\nScores+Elig\n(included)": [846, 1124, 1910, 456, 802, 860],
    "DBSCAN\nScores\n(separated)": [370, 209, 500, 1369, 1378, 456],
    "DBSCAN\nScores+Elig\n(separated)": [370, 209, 500, 1369, 1378, 456],
    "DBSCAN\nScores\n(included)": [1832, 748, 598, 456, 1766, 598],
    "DBSCAN\nScores+Elig\n(included)": [846, 1124, 1910, 456, 802, 860],
    "Agglo\nScores\n(separated)": [1143, 804, 1239, 1477, 879, 456],
    "Agglo\nScores+Elig\n(separated)": [1307, 832, 1627, 1438, 338, 456],
    "Agglo\nScores\n(included)": [1143, 1239, 804, 456, 1477, 879],
    "Agglo\nScores+Elig\n(included)": [1307, 1627, 832, 456, 1438, 338]
}

# Plot
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 25), sharey=True)
axes = axes.flatten()

for ax, (title, counts) in zip(axes, cluster_results.items()):
    ax.bar(range(len(counts)), counts, color=plt.cm.viridis(range(len(counts))))
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("User Count")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(range(len(counts)))

plt.tight_layout()
plt.suptitle("Cluster Distribution across Clustering Strategies", fontsize=16, y=1.02)
plt.show()

# Labels for each clustering variant
variant_labels = [
    "KMeans\nScores\n(separated)",
    "KMeans\nScores+Elig\n(separated)",
    "KMeans\nScores\n(included)",
    "KMeans\nScores+Elig\n(included)",
    "DBSCAN\nScores\n(separated)",
    "DBSCAN\nScores+Elig\n(separated)",
    "DBSCAN\nScores\n(included)",
    "DBSCAN\nScores+Elig\n(included)",
    "Agglo\nScores\n(separated)",
    "Agglo\nScores+Elig\n(separated)",
    "Agglo\nScores\n(included)",
    "Agglo\nScores+Elig\n(included)"
]

# Silhouette scores for each variant
silhouette_scores = [
    0.2924, 0.3257, 0.3172, 0.3699,
    0.2551, 0.1960, 0.3061, 0.2524,
    0.2910, 0.2877, 0.3391, 0.3370
]

# Davies-Bouldin Index scores for each variant
dbi_scores = [
    1.248, 1.3374, 1.1622, 1.2014,
    1.1957, 1.3506, 1.0407, 1.1894,
    1.2877, 1.3281, 1.1152, 1.1693
]

# Create horizontal bar charts for both metrics
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))

# Plot Silhouette scores
axes[0].barh(variant_labels, silhouette_scores, color="mediumseagreen")
axes[0].set_title("📈 Silhouette Score by Variant")
axes[0].set_xlabel("Silhouette Score")
axes[0].invert_yaxis()  # Highest score on top

# Plot Davies-Bouldin Index scores
axes[1].barh(variant_labels, dbi_scores, color="cornflowerblue")
axes[1].set_title("📉 Davies-Bouldin Index by Variant")
axes[1].set_xlabel("DBI (lower is better)")
axes[1].invert_yaxis()

# Adjust layout
plt.tight_layout()
plt.show()

# Extract user meta information from the session-level dataset
user_personal_data = df_cleaned.groupby("user_id").agg({
    "gender": "first",
    "age": "first",
    "age_group": "first",
    "family_status": "first",
    "membership_status": "first",
    "membership_duration_days": "max",
    "sign_up_date": "first",
}).reset_index()

# Calculate per-user booking statistics
user_aggregates = df_cleaned.groupby("user_id").agg({
    "base_fare_usd": "sum",
    "total_hotel_amount": "sum",
    "checked_bags": "sum",
    "rooms": "sum",
    "calc_nights": "mean",
    "travel_duration": "mean",
    "flight_discount_amount": "mean",
    "hotel_discount_amount": "mean",
    "discount": lambda x: (x != "no").sum(),
    "booking": "sum",
    "seats": "mean"
}).reset_index().rename(columns={
    "base_fare_usd": "total_flight_cost",
    "total_hotel_amount": "total_hotel_cost",
    "checked_bags": "total_checked_bags",
    "rooms": "total_rooms_booked",
    "calc_nights": "avg_nights_stayed",
    "travel_duration": "avg_travel_duration",
    "flight_discount_amount": "avg_flight_discount",
    "hotel_discount_amount": "avg_hotel_discount",
    "discount": "num_discounts_used",
    "booking": "num_total_bookings",
    "seats": "avg_seats_booked"
})

# Filter only sessions with bookings
df_booking_sessions = df_cleaned[df_cleaned["booking"] == True].copy()

# Determine the earliest trip start date (hotel or flight)
df_booking_sessions["trip_start_time"] = df_booking_sessions[["check_in_time", "departure_time"]].min(axis=1)

# Calculate time delta between booking session and trip start
df_booking_sessions["days_before_trip"] = (
    (df_booking_sessions["trip_start_time"] - df_booking_sessions["session_start"]).dt.total_seconds() / 86400
)
df_booking_sessions = df_booking_sessions[df_booking_sessions["days_before_trip"] >= 0]

# Compute user-level average
user_days_before_trip = (
    df_booking_sessions.groupby("user_id")["days_before_trip"]
    .mean()
    .reset_index()
    .rename(columns={"days_before_trip": "avg_days_before_trip"})
)

# Select perk score and eligibility values
perk_scores_data = df_user_profile[[
    "user_id",
    "perk_score_free_meal", "eligible_free_meal",
    "perk_score_checked_bag", "eligible_free_checked_bag",
    "perk_score_no_cancellation_fees", "eligible_no_cancellation_fees",
    "perk_score_exclusive_discounts", "eligible_exclusive_discounts",
    "perk_score_hotel_flight_combo", "eligible_free_night_combo",
    "perk_score_welcome_drink", "eligible_free_welcome_drink",
    "is_no_booker"
]]

# Extract clustering result from final model
final_clusters = df_cluster_elig[["user_id", "cluster_scores_elig_k6"]]

# Merge all data sources to create the final user table
df_user_final = (
    user_personal_data
    .merge(user_aggregates, on="user_id", how="left")
    .merge(user_days_before_trip, on="user_id", how="left")
    .merge(perk_scores_data, on="user_id", how="left")
    .merge(final_clusters, on="user_id", how="left")
)

# Basic checks after merging
print("Final DataFrame shape:", df_user_final.shape)

# Check for missing values in critical columns
print("\nMissing values in important columns:")
print(df_user_final[[
    "user_id", "gender", "age", "cluster_scores_elig_k6"
]].isnull().sum())

# Preview of a few rows
print("\nSample rows:")
display(df_user_final.sample(5))

# Final cluster distribution
print("\nFinal cluster distribution:")
print(df_user_final["cluster_scores_elig_k6"].value_counts().sort_index())

# df_user_final.to_csv("/content/drive/MyDrive/user_level_based_table_final.csv", index=False)

# Overview of cluster distribution
cluster_counts = df_user_final["cluster_scores_elig_k6"].value_counts().sort_index()
print("Cluster Distribution:")
print(cluster_counts)

# Group by cluster and compute demographic proportions
demo_cols = ["gender", "age_group", "family_status"]

for col in demo_cols:
    demo_dist = (
        df_user_final.groupby("cluster_scores_elig_k6")[col]
        .value_counts(normalize=True)
        .unstack()
        .round(2)
        .fillna(0)
    )
    print(f"\n--- {col.upper()} by Cluster ---")
    print(demo_dist)

agg_cols = [
    "total_flight_cost", "total_hotel_cost", "total_checked_bags", "total_rooms_booked",
    "avg_nights_stayed", "avg_travel_duration", "avg_flight_discount", "avg_hotel_discount",
    "num_discounts_used", "avg_days_before_trip", "num_total_bookings",
]

agg_by_cluster = df_user_final.groupby("cluster_scores_elig_k6")[agg_cols].mean().round(2)
print("\n--- Aggregated Travel Metrics by Cluster ---")
display(agg_by_cluster)

score_cols = [
    "perk_score_free_meal", "perk_score_checked_bag", "perk_score_no_cancellation_fees",
    "perk_score_exclusive_discounts", "perk_score_hotel_flight_combo", "perk_score_welcome_drink"
]

elig_cols = [
    "eligible_free_meal", "eligible_free_checked_bag", "eligible_no_cancellation_fees",
    "eligible_exclusive_discounts", "eligible_free_night_combo", "eligible_free_welcome_drink"
]

# Score averages
score_means = df_user_final.groupby("cluster_scores_elig_k6")[score_cols].mean().round(3)
print("\n--- Perk Scores by Cluster ---")
display(score_means)

# Eligibility rate (converted to binary)
elig_means = (
    df_user_final.copy()[elig_cols]
    .astype(bool)
    .astype(int)
    .groupby(df_user_final["cluster_scores_elig_k6"])
    .mean()
    .round(2)
)

print("\n--- Perk Eligibility Rate by Cluster ---")
display(elig_means)

# Function to describe a specific cluster based on user data
def describe_cluster(cluster_id, df):
    # Filter users belonging to the specific cluster
    subset = df[df["cluster_scores_elig_k6"] == cluster_id]

    # Get number of users
    size = len(subset)

    # Extract most frequent demographic values
    top_gender = subset["gender"].mode().iloc[0]
    top_age_group = subset["age_group"].mode().iloc[0]
    top_family = subset["family_status"].mode().iloc[0]

    # Compute average travel-related values
    avg_flights = subset["total_flight_cost"].mean()
    avg_hotels = subset["total_hotel_cost"].mean()
    avg_bags = subset["total_checked_bags"].mean()
    avg_rooms = subset["total_rooms_booked"].mean()
    avg_nights = subset["avg_nights_stayed"].mean()
    avg_days_before_trip = subset["avg_days_before_trip"].mean()
    num_discounts = subset["num_discounts_used"].mean()
    avg_bookings = subset["num_total_bookings"].mean()
    avg_seats = subset["avg_seats_booked"].mean()

    # Calculate average perk scores
    scores = subset[score_cols].mean().round(2)

    # Calculate average eligibility ratio per perk
    eligible = (
        subset[eligibility_cols]
        .astype(bool)
        .astype(int)
        .mean()
        .round(2)
    )

    # Generate and print a cluster summary
    print(f"\nCluster {cluster_id} – {size} users")
    print(f"- Most are {top_gender.lower()}, aged {top_age_group}, family status: {top_family}")
    print(f"- Avg. flight cost: $ {avg_flights:.0f}, hotel cost: $ {avg_hotels:.0f}")
    print(f"- Avg. checked bags: {avg_bags:.1f}, rooms booked: {avg_rooms:.1f}")
    print(f"- Avg. nights stayed: {avg_nights:.1f}, days booked in advance: {avg_days_before_trip:.1f}")
    print(f"- Avg. discounts used: {num_discounts:.1f}")
    print(f"- Avg. total bookings: {avg_bookings:.1f}, seats booked: {avg_seats:.1f}")
    print(f"- Perk Scores:\n{scores}")
    print(f"- Perk Eligibility:\n{eligible}")

# Loop over all unique cluster labels and describe each
for cluster in sorted(df_user_final["cluster_scores_elig_k6"].dropna().unique()):
    describe_cluster(cluster, df_user_final)

# Mapping: cluster_id → perk score column
cluster_perk_map = {
    0: "exclusive_discounts",           # All get it (no score)
    1: "checked_bag",                   # Needs score threshold
    2: "no_cancellation_fees",          # Needs score threshold
    3: "welcome_drink",                 # All get it (no score)
    4: "free_meal",                     # Needs score threshold
    5: "hotel_flight_combo"             # Needs score threshold
}

# Optional thresholds (only for clusters needing score filtering)
custom_thresholds = {
    1: 0.75,
    2: 0.50,
    4: 0.75,
    5: 0.75
}

# Reset column
df_user_final["assigned_perk"] = None

# Assign perks
for cluster, perk in cluster_perk_map.items():
    mask = df_user_final["cluster_scores_elig_k6"] == cluster

    # Only apply threshold if defined
    if cluster in custom_thresholds:
        score_col = f"perk_score_{perk}"
        threshold = df_user_final.loc[mask, score_col].quantile(custom_thresholds[cluster])
        mask = mask & (df_user_final[score_col] >= threshold)

    # Assign perk
    df_user_final.loc[mask, "assigned_perk"] = perk

print("\nPerk Distribution by Cluster:")
display(
    pd.crosstab(df_user_final["cluster_scores_elig_k6"], df_user_final["assigned_perk"])
)

# Mapping: cluster_id → marketing-friendly group name
marketing_names = {
    0: "The Young Explorers",
    1: "The Frequent Flyers",
    2: "The Savvy Planners",
    3: "The Curious Visitors",
    4: "The Premium Travelers",
    5: "The Spontaneous Stayers"
}

# Create a new column with marketing name
df_user_final["perk_group_marketing_name"] = df_user_final["cluster_scores_elig_k6"].map(marketing_names)

df_user_final.info()

# df_user_final.to_csv("/content/drive/MyDrive/user_level_based_table_clustered_final.csv", index=False)

# Define base columns for user info and clusters
base_cols = [
    "user_id", "cluster_scores_elig_k6", "assigned_perk", "perk_group_marketing_name",
    "gender", "age", "age_group", "family_status",
    "total_flight_cost", "total_hotel_cost", "total_checked_bags", "total_rooms_booked",
    "avg_nights_stayed", "avg_days_before_trip", "num_discounts_used",
    "num_total_bookings", "avg_seats_booked"
]

# Perk score and eligibility columns
score_cols = [
    "perk_score_free_meal", "perk_score_checked_bag", "perk_score_no_cancellation_fees",
    "perk_score_exclusive_discounts", "perk_score_hotel_flight_combo", "perk_score_welcome_drink"
]

eligibility_cols = [
    "eligible_free_meal", "eligible_free_checked_bag", "eligible_no_cancellation_fees",
    "eligible_exclusive_discounts", "eligible_free_night_combo", "eligible_free_welcome_drink"
]

# Combine all relevant columns
cols_to_keep = base_cols + score_cols + eligibility_cols

# Filter only users with an assigned perk
df_perk_recipients = df_user_final[df_user_final["assigned_perk"].notna()].copy()

# Subset the DataFrame
df_perk_recipients_export = df_perk_recipients[cols_to_keep]

# Export to CSV
# df_perk_recipients_export.to_csv("/content/drive/MyDrive/user_perk_assignment_final.csv", index=False)
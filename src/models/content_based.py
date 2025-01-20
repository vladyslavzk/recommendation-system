from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """Content-based recommendation model using product features."""

    def __init__(self, spark_session: SparkSession):
        """Initialize the recommender model."""
        self.spark = spark_session
        self.product_features = None
        self.similarity_matrix = None
        self.product_idx_mapping = None
        self.idx_product_mapping = None

    def fit(self, df) -> None:

        df.cache()

        print("Number of rows in DataFrame:", df.count())

        # Convert features to numpy array using RDD operations
        product_features_rdd = df.select('product_id', 'features').rdd.map(
            lambda row: (row.product_id, row.features.toArray())
        ).collect()

        print("Number of items collected:", len(product_features_rdd))
        print("First few items:", product_features_rdd[:3])

        product_ids, features = zip(*product_features_rdd)

        print(f"Number of unique product_ids: {len(set(product_ids))}")

        # Get product IDs
        self.product_idx_mapping = {
            pid: idx for idx, pid in enumerate(product_ids)
        }
        self.idx_product_mapping = {
            idx: pid for pid, idx in self.product_idx_mapping.items()
        }

        print("Number of items in mappings:", len(self.product_idx_mapping))
        print("First few mapping items:", list(self.product_idx_mapping.items())[:3])

        print("Number of items in mappings:", len(self.idx_product_mapping))
        print("First few mapping idx:", list(self.idx_product_mapping.items())[:3])

        self.product_features = np.array(features)

        # Calculate similarity matrix
        logger.info("Calculating similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.product_features)


    def get_similar_products(self,
                             product_id: int,
                             n: int = 5,
                             exclude_viewed: List[int] = None) -> List[Tuple[int, float]]:
        """
        Get n most similar products for a given product.

        Args:
            product_id: ID of the product to find similarities for
            n: Number of similar products to return
            exclude_viewed: List of product IDs to exclude (e.g., already viewed)

        Returns:
            List of tuples (product_id, similarity_score)
        """
        if product_id not in self.product_idx_mapping:
            logger.warning(f"Product ID {product_id} not found in training data")
            return []

        # Get product index
        idx = self.product_idx_mapping[product_id]

        # Get similarity scores for this product
        similarities = self.similarity_matrix[idx]

        # Create list of (idx, similarity) pairs and sort
        product_similarities = list(enumerate(similarities))
        product_similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter out the query product and excluded products
        exclude_viewed = set(exclude_viewed) if exclude_viewed else set()
        similar_products = []

        for idx, score in product_similarities[1:]:  # Skip the first one (self-similarity)
            product_id = self.idx_product_mapping[idx]
            if product_id not in exclude_viewed:
                similar_products.append((product_id, float(score)))
                if len(similar_products) >= n:
                    break

        return similar_products

    def get_recommendations(self,
                            user_history: List[int],
                            n: int = 5) -> List[Tuple[int, float]]:
        """
        Get recommendations based on user's viewing history.

        Args:
            user_history: List of product IDs the user has interacted with
            n: Number of recommendations to return

        Returns:
            List of tuples (product_id, score)
        """
        if not user_history:
            logger.warning("Empty user history provided")
            return []

        # Get similar products for each item in user history
        all_similarities = []
        for product_id in user_history:
            if product_id in self.product_idx_mapping:
                similar_products = self.get_similar_products(
                    product_id,
                    n=n,
                    exclude_viewed=user_history
                )
                all_similarities.extend(similar_products)

        # Aggregate scores for same products
        product_scores = {}
        for product_id, score in all_similarities:
            if product_id not in product_scores:
                product_scores[product_id] = score
            else:
                # Use max score when product appears multiple times
                product_scores[product_id] = max(product_scores[product_id], score)

        # Sort by score and return top n
        recommendations = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return recommendations[:n]

    def explain_recommendation(self,
                               product_id: int,
                               recommended_id: int) -> Dict:
        """
        Explain why a product was recommended.

        Args:
            product_id: Original product ID
            recommended_id: Recommended product ID

        Returns:
            Dictionary with explanation details
        """
        if (product_id not in self.product_idx_mapping or
                recommended_id not in self.product_idx_mapping):
            return {"error": "Product not found"}

        # Get feature vectors
        idx1 = self.product_idx_mapping[product_id]
        idx2 = self.product_idx_mapping[recommended_id]
        features1 = self.product_features[idx1]
        features2 = self.product_features[idx2]

        # Calculate feature-wise similarities
        feature_similarities = {
            "category_similarity": cosine_similarity(
                features1[:32].reshape(1, -1),
                features2[:32].reshape(1, -1)
            )[0][0],
            "price_similarity": 1 - abs(features1[32] - features2[32]),
            "popularity_similarity": cosine_similarity(
                features1[33:].reshape(1, -1),
                features2[33:].reshape(1, -1)
            )[0][0]
        }

        return feature_similarities
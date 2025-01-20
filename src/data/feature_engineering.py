import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, FloatType

from pyspark.ml.feature import VectorAssembler

from gensim.models import Word2Vec
import networkx as nx

import numpy as np
from sklearn.preprocessing import StandardScaler

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFeatures:
    """Base class for feature engineering with shared utilities."""

    def __init__(self, spark_session: SparkSession):
        """
        Initialize the base feature engineering class.

        Args:
            spark_session: Active Spark session for data processing
        """
        self.spark = spark_session

    def _normalize_column(self, df, column_name: str, output_name: str = None):
        """
        Normalize values in a column to range [0,1].

        Args:
            df: Spark DataFrame
            column_name: Name of column to normalize
            output_name: Name for normalized column (defaults to column_name + '_normalized')

        Returns:
            DataFrame with normalized column added
        """
        output_name = output_name or f"{column_name}_normalized"

        # Calculate min and max values
        stats = df.select(
            F.min(column_name).alias('min'),
            F.max(column_name).alias('max')
        ).collect()[0]

        # Normalize the column
        return df.withColumn(
            output_name,
            (F.col(column_name) - stats['min']) / (stats['max'] - stats['min'])
        )


class ContentBasedFeatures(BaseFeatures):
    """Generate features based on product content and metadata."""
    def __init__(self, spark, vector_size=32):
        super().__init__(spark)
        self._vector_size = vector_size
        self._G = nx.Graph()

    def _build_graph(self, df: pyspark.sql.DataFrame):
        brands = df.select('brand').distinct()

        self._G.add_nodes_from(brands)

        import networkx as nx

        # Create graph
        G = nx.Graph()

        # Add nodes (brands)
        brands = ["Samsung", "Apple", "LG", "Nike", "Adidas", "Beko", "Rolex"]
        G.add_nodes_from(brands)
        import networkx as nx

        # Create graph
        G = nx.Graph()

        # Add nodes (brands)
        brands = ["Samsung", "Apple", "LG", "Nike", "Adidas", "Beko", "Rolex"]
        G.add_nodes_from(brands)

        # Add edges with weights
        edges = [
            ("Samsung", "Apple", 0.9),  # Strong relationship
            ("Samsung", "LG", 0.6),  # Medium relationship
            ("Nike", "Adidas", 0.8),  # Strong relationship
            ("Samsung", "Beko", 0.1),  # Weak relationship
            ("Rolex", "Beko", 0)  # No relationship
        ]
        G.add_weighted_edges_from(edges)




    def create_category_embeddings(self, df: pyspark.sql.DataFrame):
        """
        Create category embeddings using weighted paths based on interaction types.

        Args:
            df: Spark DataFrame with category_code and event_type columns
            vector_size: Dimension of embedding vectors

        Returns:
            DataFrame with category embeddings added
        """

        # TODO: Poincare Model instead of Word2Vec

        logger.info("Creating weighted category paths...")

        # Create weighted paths based on event type
        df_paths = df.withColumn('levels', F.split('category_code', '\\.'))

        # Apply weights based on interaction type
        df_weighted = df_paths.withColumn('weight',
                                          F.when(F.col('event_type') == 'purchase', 3)
                                          .when(F.col('event_type') == 'cart', 2)
                                          .otherwise(1))

        # Generate hierarchical paths efficiently
        df_with_paths = df_weighted.withColumn('paths',
                                               F.transform(
                                                   F.sequence(F.lit(1), F.size('levels')),
                                                   lambda x: F.concat_ws('.', F.slice('levels', 1, x))
                                               ))

        # Create final weighted paths
        final_paths = df_with_paths.select(
            F.explode(
                F.flatten(
                    F.array_repeat('paths', 'weight')
                )
            ).alias('category_path')
        )

        logger.info("Training Word2Vec model...")
        # Train Word2Vec model
        paths = final_paths.collect()
        model = Word2Vec(
            sentences=paths,
            vector_size=self._vector_size,
            window=5,
            min_count=1,
            workers=4,
            sg=1
        )

        # Create UDF for embedding lookup
        category_vectors = [(word, model.wv[word].tolist())
                            for word in model.wv.index_to_key]

        # Create DataFrame with embeddings
        embedding_df = self.spark.createDataFrame(
            category_vectors,
            ["category_code", "category_embedding"]
        )

        embedding_df.filter(embedding_df.category_embedding.isNull()).show(5)

        logger.info("Adding embedding features to DataFrame...")
        return df.join(
            embedding_df,
            on='category_code',
            how='left'
        )



    def create_price_features(self, df: pyspark.sql.DataFrame):
        """
        Create price-based features including normalized prices and price brackets.

        Args:
            df: Spark DataFrame with price column

        Returns:
            DataFrame with price features added
        """
        logger.info("Creating price-based features...")

        # Can be extended to
        # 1. average across a time window,
        # 2. weighted average price,
        # 3. use the last price occurrence
        price_metrics = df.groupBy('product_id').agg(
            F.avg(F.col('price')).alias('average_price'),
            F.stddev(F.col('price')).alias('price_volatility'),
        )

        # Create price brackets
        price_quantiles = price_metrics.approxQuantile("average_price", [0.2, 0.4, 0.6, 0.8], 0.01)
        price_metrics = price_metrics.withColumn('price_bracket',
                           F.when(F.col('average_price') <= price_quantiles[0], 'very_low')
                           .when(F.col('average_price') <= price_quantiles[1], 'low')
                           .when(F.col('average_price') <= price_quantiles[2], 'medium')
                           .when(F.col('average_price') <= price_quantiles[3], 'high')
                           .otherwise('very_high')
                           )

        price_metrics = self._normalize_column(price_metrics, 'average_price')
        price_metrics = self._normalize_column(price_metrics, 'price_volatility')

        return df.join(price_metrics, 'product_id')

    # don't need them in content-based filtering
    def create_popularity_features(self, df: pyspark.sql.DataFrame):
        """
        Create popularity-based features for products.

        Args:
            df: Spark DataFrame with product interactions

        Returns:
            DataFrame with popularity features added
        """
        logger.info("Creating popularity features...")

        # Calculate various popularity metrics
        popularity_metrics = df.groupBy('product_id').agg(
            F.count('*').alias('total_interactions'),
            F.sum(F.when(F.col('event_type') == 'view', 1).otherwise(0)).alias('view_count'),
            F.sum(F.when(F.col('event_type') == 'cart', 1).otherwise(0)).alias('cart_count'),
            F.sum(F.when(F.col('event_type') == 'purchase', 1).otherwise(0)).alias('purchase_count')
        )

        # Calculate conversion rates
        popularity_metrics = popularity_metrics.withColumn(
            'cart_to_view_rate',
            F.when(
                F.col('view_count') > 0,
                F.col('cart_count') / F.col('view_count')
                ).otherwise(0.0)
        ).withColumn(
            'purchase_to_cart_rate',
            F.when(
                F.col('cart_count') > 0,
                F.col('purchase_count') / F.col('cart_count')
                ).otherwise(0.0)
        )

        return df.join(popularity_metrics, 'product_id')

    def create_features(self, df: pyspark.sql.DataFrame):
        df = self.create_category_embeddings(df)
        df = self.create_price_features(df)

        # Aggregate features at product level
        df_aggregated = df.groupBy('product_id').agg(
            # Category embeddings (already at product level)
            F.first('category_embedding').alias('category_embedding'),
            # Price features
            F.first('average_price_normalized').alias('price_normalized')
        )
        for i in range(self._vector_size):
            df_aggregated = df_aggregated.withColumn(
                f'category_emb_{i}',
                F.element_at('category_embedding', i + 1)
            )

        # List all numeric features we want to use
        feature_cols = (
                [f'category_emb_{i}' for i in range(self._vector_size)] +
                ['price_normalized']
        )
        print(f"Feature columns: {feature_cols}")
        print(f"Columns in df_aggregated: {df_aggregated.columns}")
        # Combine features into a single vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol='features'
        )
        return assembler.transform(df_aggregated)



class CollaborativeFeatures(BaseFeatures):
    """Generate features based on user-item interactions."""

    def create_interaction_matrix(self, df: pyspark.sql.DataFrame):
        """
        Create user-item interaction matrix with weighted interactions.

        Args:
            df: Spark DataFrame with user-item interactions

        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction matrix...")

        # Create weighted interaction values
        interaction_weights = df.withColumn('interaction_weight',
                                            F.when(F.col('event_type') == 'purchase', 3.0)
                                            .when(F.col('event_type') == 'cart', 2.0)
                                            .when(F.col('event_type') == 'view', 1.0)
                                            .otherwise(0.0))

        # Aggregate interactions
        return interaction_weights.groupBy('user_id', 'product_id').agg(
            F.sum('interaction_weight').alias('interaction_strength')
        )

    def create_user_features(self, df: pyspark.sql.DataFrame):
        """
        Create user behavior features.

        Args:
            df: Spark DataFrame with user interactions

        Returns:
            DataFrame with user features added
        """
        logger.info("Creating user behavior features...")

        user_features = df.groupBy('user_id').agg(
            F.count('*').alias('total_interactions'),
            F.countDistinct('product_id').alias('unique_products'),
            F.countDistinct('category_code').alias('unique_categories'),
            F.avg('price').alias('avg_price_interaction'),
            F.stddev('price').alias('price_std'),
            F.max('price').alias('max_price_interaction')
        )

        return df.join(user_features, 'user_id')


class HybridFeatures:
    """Combine and process features from multiple sources."""

    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.content_features = ContentBasedFeatures(spark_session)
        self.collab_features = CollaborativeFeatures(spark_session)

    def generate_all_features(self, df: pyspark.sql.DataFrame):
        """
        Generate all features for hybrid recommendation.

        Args:
            df: Input Spark DataFrame

        Returns:
            DataFrame with all features added
        """
        logger.info("Generating all features for hybrid recommendation...")

        # Generate content-based features
        df = self.content_features.create_category_embeddings(df)
        df = self.content_features.create_price_features(df)
        df = self.content_features.create_popularity_features(df)

        # Generate collaborative features
        df = self.collab_features.create_user_features(df)
        interaction_matrix = self.collab_features.create_interaction_matrix(df)

        # Combine with interaction matrix
        df = df.join(interaction_matrix, ['user_id', 'product_id'], 'left')

        return df


# Usage example:
if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

    # Create feature generator
    feature_generator = HybridFeatures(spark)

    # Generate features
    df_with_features = feature_generator.generate_all_features(input_df)
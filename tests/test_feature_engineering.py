import pytest
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from src.data import (
    ContentBasedFeatures,
    CollaborativeFeatures,
    HybridFeatures
)

@pytest.fixture(scope="session")
def spark():
    import os
    os.environ['PYSPARK_PYTHON'] = 'I:/learning_projects/environments/ecommerce_venv/Scripts/python.exe'
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'I:/learning_projects/environments/ecommerce_venv/Scripts/python.exe'

    """Create a Spark session for testing."""
    return (SparkSession.builder
            .appName("FeatureEngineeringTests")
            .master("local[*]")
            .getOrCreate())


@pytest.fixture
def sample_data(spark):
    """Create sample data for testing."""
    # Create a small dataset that represents your real data
    data = [
        # user_id, product_id, category_code, price, event_type
        (1, 101, "electronics.smartphone", 500.0, "view"),
        (1, 101, "electronics.smartphone", 500.0, "purchase"),
        (2, 102, "electronics.laptop", 1000.0, "view"),
        (2, 103, "electronics.accessories", 50.0, "cart"),
    ]

    return spark.createDataFrame(
        data,
        ["user_id", "product_id", "category_code", "price", "event_type"]
    )


class TestContentBasedFeatures:
    """Test content-based feature engineering."""

    def test_category_embeddings(self, spark, sample_data):
        """Test that category embeddings are created correctly."""
        content_features = ContentBasedFeatures(spark)
        result_df = content_features.create_category_embeddings(sample_data)

        # Check that embeddings column exists
        assert 'category_embedding' in result_df.columns

        # Check embedding dimension
        first_embedding = result_df.select('category_embedding').first()[0]
        assert len(first_embedding) == 32  # Default embedding size

        # Check that all rows have embeddings
        null_embeddings = result_df.filter(
            F.col('category_embedding').isNull()
        ).count()
        assert null_embeddings == 0

    def test_price_features(self, spark, sample_data):
        """Test price-based feature creation."""
        content_features = ContentBasedFeatures(spark)
        result_df = content_features.create_price_features(sample_data)

        # Check that new columns exist
        assert 'price_normalized' in result_df.columns
        assert 'price_bracket' in result_df.columns

        # Check normalization range
        price_stats = result_df.select(
            F.min('price_normalized').alias('min'),
            F.max('price_normalized').alias('max')
        ).first()

        assert price_stats['min'] >= 0.0
        assert price_stats['max'] <= 1.0


class TestCollaborativeFeatures:
    """Test collaborative feature engineering."""

    def test_interaction_matrix(self, spark, sample_data):
        """Test interaction matrix creation."""
        collab_features = CollaborativeFeatures(spark)
        result_df = collab_features.create_interaction_matrix(sample_data)

        # Check that interaction strength exists
        assert 'interaction_strength' in result_df.columns

        # Verify weights are correct
        # Find interaction strength for user 1, product 101
        # Should be 3.0 (purchase) + 1.0 (view) = 4.0
        strength = result_df.filter(
            (F.col('user_id') == 1) &
            (F.col('product_id') == 101)
        ).first()['interaction_strength']

        assert strength == 4.0


class TestHybridFeatures:
    """Test hybrid feature generation."""

    def test_generate_all_features(self, spark, sample_data):
        """Test complete feature generation pipeline."""
        hybrid_features = HybridFeatures(spark)
        result_df = hybrid_features.generate_all_features(sample_data)

        # Check that we have features from both approaches
        expected_columns = {
            'category_embedding',
            'price_normalized',
            'price_bracket',
            'interaction_strength',
            'total_interactions'
        }

        assert all(col in result_df.columns for col in expected_columns)

        # Check data quality
        assert result_df.count() == sample_data.count()
        assert not result_df.filter(F.col('price_normalized').isNull()).count()


def test_logging_configuration(caplog):
    """Test that logging is configured correctly."""
    # Create features with logging
    spark = SparkSession.builder.getOrCreate()
    features = ContentBasedFeatures(spark)

    # Generate some features to trigger logs
    sample_data = spark.createDataFrame(
        [(1, "test", 10.0)],
        ["id", "category", "price"]
    )

    features.create_price_features(sample_data)

    # Check that logs were created
    assert "Creating price-based features" in caplog.text
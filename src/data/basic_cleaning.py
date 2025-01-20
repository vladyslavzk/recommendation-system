import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicCleaner:
    def _duplicate_removal(self, df: pyspark.sql.DataFrame):
        logger.info("Removing duplicates from DataFrame...")

        df = df.dropDuplicates(['event_type', 'product_id',
                                'category_id', 'category_code',
                                'brand', 'price',
                                'user_id', 'user_session'])
        return df

    def _price_filtering(self, df: pyspark.sql.DataFrame):
        logger.info("Price filtering...")
        df = df.filter(df.price > 0)
        return df

    def _handle_missing_category_code(self, df):
        logger.info("Handling missing category_code...")

        df = df.withColumn('category_code',
                           F.when(
                               F.col('category_code').isNull(),
                               F.concat(F.lit('custom_category_'), F.col('category_id').cast('string'))
                           ).otherwise(F.col('category_code')))
        return df

    def _handle_missing_brands(self, df):
        logger.info("Handling missing brands...")

        df = df.withColumn('brand',
                           F.when(
                               F.col('brand').isNull(),
                               'unknown'
                           ).otherwise(F.col('brand')))
        return df

    def clean_df(self, df):
        logger.info("Cleaning DataFrame...")

        df = self._duplicate_removal(df)
        df = self._price_filtering(df)
        df = self._handle_missing_category_code(df)
        df = self._handle_missing_brands(df)

        return df

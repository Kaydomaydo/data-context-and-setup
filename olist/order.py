import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        orders = self.data['orders'].copy()

        orders= orders[orders['order_status'] == 'delivered'].copy() 
        orders['order_purchase_timestamp']=pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date']=pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date']=pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['wait_time']=(orders['order_delivered_customer_date']-orders['order_purchase_timestamp'])/np.timedelta64(1,'D')
        orders['expected_wait_time']= (orders['order_estimated_delivery_date']-orders['order_purchase_timestamp'])/np.timedelta64(1,'D')
        orders['delay_vs_expected']= (orders['order_delivered_customer_date']-orders['order_estimated_delivery_date'])/np.timedelta64(1,'D')
        orders['delay_vs_expected']=orders['delay_vs_expected'].clip(lower=0)
        get_wait_time_results=orders[['order_id','wait_time','expected_wait_time','delay_vs_expected','order_status']]

        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        reviews['dim_is_five_star'] = (reviews['review_score'] == 5).astype(int)
        reviews['dim_is_one_star'] = (reviews['review_score'] == 1).astype(int)

        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        order_items = self.data['order_items'].copy()
        get_number_items_results = order_items.groupby('order_id', as_index=False).agg(
        number_of_items=('order_item_id', 'count')
         )
        return get_number_items_results

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        order_items = self.data['order_items']
        result = order_items.groupby('order_id', as_index=False).agg(number_of_sellers=('seller_id', 'nunique'))
        return result

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        order_items = self.data['order_items']
        result = order_items.groupby('order_id', as_index=False).agg(
            price=('price', 'sum'),
            freight_value=('freight_value', 'sum')
        )
        return result

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        data = self.data
        order_items = data['order_items']
        orders = data['orders']
        sellers = data['sellers']
        customers = data['customers']
        geo = data['geolocation']

        # Keep one geo entry per zip code prefix (median lat/lon)
        geo = geo.groupby('geolocation_zip_code_prefix', as_index=False).agg(
            geolocation_lat=('geolocation_lat', 'median'),
            geolocation_lng=('geolocation_lng', 'median')
        )

        # Seller location
        sellers_geo = sellers.merge(
            geo, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix'
        )[['seller_id', 'geolocation_lat', 'geolocation_lng']]\
          .rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'})

        # Customer location
        customers_geo = customers.merge(
            geo, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix'
        )[['customer_id', 'geolocation_lat', 'geolocation_lng']]\
          .rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})

        # Join everything
        df = order_items[['order_id', 'seller_id']]\
            .merge(orders[['order_id', 'customer_id']], on='order_id')\
            .merge(sellers_geo, on='seller_id')\
            .merge(customers_geo, on='customer_id')

        df['distance_seller_customer'] = df.apply(
            lambda row: haversine_distance(
                row['seller_lng'], row['seller_lat'],
                row['customer_lng'], row['customer_lat']
            ), axis=1
        )

        return df.groupby('order_id', as_index=False).agg(
            distance_seller_customer=('distance_seller_customer', 'mean')
        )

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_items', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # Hint: make sure to re-use your instance methods defined above
        df = self.get_wait_time(is_delivered)\
            .merge(self.get_review_score(), on='order_id')\
            .merge(self.get_number_items(), on='order_id')\
            .merge(self.get_number_sellers(), on='order_id')\
            .merge(self.get_price_and_freight(), on='order_id')

        if with_distance_seller_customer:
            df = df.merge(self.get_distance_seller_customer(), on='order_id')

        return df.dropna()

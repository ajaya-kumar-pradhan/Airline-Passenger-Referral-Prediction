import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_mock_data():
    num_customers = 500
    num_orders = 2000
    
    # --- Generate Customers (EcommerceDataset2.xlsx) ---
    customer_ids = [f"CUST{i:04d}" for i in range(1, num_customers + 1)]
    countries = ['USA', 'UK', 'Canada', 'Germany', 'France', 'Australia']
    memberships = ['Standard', 'Premium', 'VIP']
    
    customers_data = {
        'CustomerID': customer_ids,
        'CustomerEmail': [f"user{i}@example.com" for i in range(1, num_customers + 1)],
        'Country': [random.choice(countries) for _ in range(num_customers)],
        'Membership': [random.choices(memberships, weights=[0.6, 0.3, 0.1])[0] for _ in range(num_customers)],
        'SignUpDate': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1000)) for _ in range(num_customers)],
        'TotalSpent': [0 for _ in range(num_customers)] # Placeholder, can be filled later
    }
    
    # Calculate LastOrderDate later based on orders
    customers_df = pd.DataFrame(customers_data)
    
    # --- Generate Orders (EcommerceDataset1.xlsx) ---
    products = {
        'Laptop': 1200, 'Smartphone': 800, 'Headphones': 150, 'Monitor': 300, 
        'Keyboard': 50, 'Mouse': 30, 'Tablet': 500, 'Charger': 20
    }
    product_list = list(products.keys())
    
    orders_data = {
        'OrderID': [f"ORD{i:05d}" for i in range(1, num_orders + 1)],
        'CustomerID': [random.choice(customer_ids) for _ in range(num_orders)],
        'Product': [random.choice(product_list) for _ in range(num_orders)],
        'Quantity': [random.randint(1, 5) for _ in range(num_orders)],
        'OrderDate': [datetime(2021, 1, 1) + timedelta(days=random.randint(0, 700)) for _ in range(num_orders)],
        'ShippingCost': [random.uniform(5, 50) for _ in range(num_orders)]
    }
    
    orders_df = pd.DataFrame(orders_data)
    orders_df['UnitPrice'] = orders_df['Product'].map(products)
    
    # --- Align Data ---
    # Update LastOrderDate and TotalSpent in Customers based on Orders
    
    customer_aggregates = orders_df.groupby('CustomerID').agg({
        'OrderDate': 'max',
        'UnitPrice': 'sum' # Approximation, realistically needs (Qty * Price) sum
    }).reset_index()
    
    # Correct calculation for TotalSpent
    orders_df['RowTotal'] = orders_df['Quantity'] * orders_df['UnitPrice']
    real_spend = orders_df.groupby('CustomerID')['RowTotal'].sum().reset_index()
    real_spend.columns = ['CustomerID', 'RealTotalSpent']
    
    last_dates = orders_df.groupby('CustomerID')['OrderDate'].max().reset_index()
    last_dates.columns = ['CustomerID', 'RealLastOrderDate']
    
    customers_df = pd.merge(customers_df, real_spend, on='CustomerID', how='left').fillna(0)
    customers_df = pd.merge(customers_df, last_dates, on='CustomerID', how='left')
    
    customers_df['TotalSpent'] = customers_df['RealTotalSpent']
    customers_df['LastOrderDate'] = customers_df['RealLastOrderDate']
    
    # Handle customers with no orders (if any)
    customers_df['LastOrderDate'] = customers_df['LastOrderDate'].fillna(customers_df['SignUpDate'])
    
    # Drop temp columns
    customers_df = customers_df.drop(columns=['RealTotalSpent', 'RealLastOrderDate'])
    orders_df = orders_df.drop(columns=['RowTotal'])

    # --- Save ---
    # Ensure directory exists (it should)
    orders_df.to_excel("ecommerce_analytics/data/EcommerceDataset1.xlsx", index=False)
    customers_df.to_excel("ecommerce_analytics/data/EcommerceDataset2.xlsx", index=False)
    
    print("Mock data generated successfully.")

if __name__ == "__main__":
    generate_mock_data()

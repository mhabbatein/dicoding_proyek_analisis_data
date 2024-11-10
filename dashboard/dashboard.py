import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule="D", on="order_purchase_timestamp").agg(
        {"order_id": "nunique", "price": "sum"}
    )
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(
        columns={"order_id": "order_count", "price": "revenue"}, inplace=True
    )
    return daily_orders_df


def create_sum_order_items_df(df):
    sum_order_items_df = (
        df.groupby("product_category_name")["order_item_id"]
        .count()
        .sample(5)
        .sort_values(ascending=False)
        .reset_index()
    )
    sum_order_items_df.rename(columns={"order_item_id": "quantity_sold"}, inplace=True)

    return sum_order_items_df


def create_customer_count_by_state_df(df):
    customer_count_by_state_df = (
        df.groupby("customer_state")["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    customer_count_by_state_df.rename(
        columns={"customer_id": "customer_count"}, inplace=True
    )
    return customer_count_by_state_df

def create_rfm_df(all_df):
    reference_date = all_df['order_purchase_timestamp'].max()

    rfm_df = all_df.groupby('customer_unique_id').agg({
        # Recency:  reference date - latest purchase
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        
        # Frequency: Count of unique orders by each customer
        'order_id': 'nunique',
        
        # Monetary: Sum of all purchases for each customer
        'price': 'sum'
        
    }).reset_index()

    rfm_df.rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    }, inplace=True)

    rfm_df['r_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['Monetary'].rank(ascending=True)

    rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*100
    
    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    rfm_df['RFM_score'] = 0.15*rfm_df['r_rank_norm']+0.28 * \
    rfm_df['f_rank_norm']+0.57*rfm_df['m_rank_norm']
    rfm_df['RFM_score'] *= 0.05
    rfm_df = rfm_df.round(2)
    rfm_df[['customer_unique_id', 'RFM_score']].head(5).sort_values(by="RFM_score", ascending=False)
    
    rfm_df["customer_segment"] = np.where(
    rfm_df['RFM_score'] > 4.5, "Top customers", (np.where(
        rfm_df['RFM_score'] > 4, "High value customer",(np.where(
            rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
                rfm_df['RFM_score'] > 1.6, 'Low value customers', 'lost customers'))))))
    customer_segment_df = rfm_df.groupby(by="customer_segment", as_index=False).customer_unique_id.nunique()

    customer_segment_df['customer_segment'] = pd.Categorical(customer_segment_df['customer_segment'], [
    "lost customers", "Low value customers", "Medium value customer",
    "High value customer", "Top customers"])

    return rfm_df, customer_segment_df



def display_visualisasi_pertama(df):
    st.header("Pertanyaan 1")
    st.write("Bagaimana distribusi frekuensi pesanan berdasarkan status pesanan (order_status)?")
    status_counts = df["order_status"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(x=status_counts.index, height=status_counts.values, color="skyblue")
    
    ax.set_title("Distribusi Frekuensi Pesanan berdasarkan Status Pemesanan")
    ax.set_xlabel("Status Pemesanan")
    ax.set_ylabel("Frekuensi")
    
    ax.set_xticks(range(len(status_counts.index)))
    ax.set_xticklabels(status_counts.index, rotation=45)
    
    st.pyplot(fig)
    st.dataframe(status_counts)

def display_visualisasi_kedua(df):
    st.header("Pertanyaan 2")
    st.write("Bagaimana rata-rata waktu yang dibutuhkan dari pembelian hingga pengiriman ke pelanggan (dari order_purchase_timestamp hingga order_delivered_customer_date)?")
    delivery_time = df['delivery_time']
    fig, ax = plt.subplots()
    ax.hist(delivery_time, bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Distribusi Waktu Delivery")
    ax.set_xlabel("Waktu Delivery (hari)")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

def display_sum_order_items(sum_order_items_df, customer_count_by_state_df):
    st.header("Top-Selling Products")
    st.write("Jumlah Produk Terjual.")
    st.dataframe(sum_order_items_df)

    fig, ax = plt.subplots()
    sum_order_items_df.plot(
        kind="bar", x="product_category_name", y="quantity_sold", ax=ax, legend=False
    )
    ax.set_ylabel("Quantity Sold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    st.header("Customer Count by State")
    st.write("Jumlah pelanggan berdasarkan negara.")

    fig, ax = plt.subplots(figsize=(10, 6))
    customer_count_by_state_df.plot(
        kind="bar", x="customer_state", y="customer_count", ax=ax, legend=False
    )
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)


def display_rfm(rfm_df):
    st.header("RFM Analysis")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

    colors = ["#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4", "#72BCD4"]

    sns.barplot(y="Recency", x="customer_unique_id", data=rfm_df.sort_values(by="Recency", ascending=True).head(5), palette=colors, ax=ax[0], hue="Recency")
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("By Recency (days)", loc="center", fontsize=18)
    ax[0].tick_params(axis ='x', labelsize=15, rotation=45)

    sns.barplot(y="Frequency", x="customer_unique_id", data=rfm_df.sort_values(by="Frequency", ascending=False).head(5), palette=colors, ax=ax[1], hue="Frequency")
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("By Frequency", loc="center", fontsize=18)
    ax[1].tick_params(axis='x', labelsize=15, rotation=45)

    sns.barplot(y="Monetary", x="customer_unique_id", data=rfm_df.sort_values(by="Monetary", ascending=False).head(5), palette=colors, ax=ax[2], hue="Monetary")
    ax[2].set_ylabel(None)
    ax[2].set_xlabel(None)
    ax[2].set_title("By Monetary", loc="center", fontsize=18)
    ax[2].tick_params(axis='x', labelsize=15, rotation=45)

    plt.suptitle("Best Customer Based on RFM Parameters (customer_unique_id)", fontsize=20)
    st.pyplot(fig)

def display_customer_segment(customer_segment_df):
    st.dataframe(customer_segment_df)
    st.header("Customer Segments")
    # Create a new figure
    fig = plt.figure(figsize=(10, 5))
    colors_ = ["#72BCD4", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    
    # Plot the data
    sns.barplot(
        x="customer_unique_id", 
        y="customer_segment",
        data=customer_segment_df.sort_values(by="customer_segment", ascending=False),
        palette=colors_,
        orient = 'h'
    )
    
    # Add labels and title
    plt.title("Number of Customers for Each Segment", loc="center", fontsize=15)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.tick_params(axis='y', labelsize=12, rotation=45)
    
    # Display the plot in Streamlit
    st.pyplot(fig)


# Load cleaned data
all_df = pd.read_csv("./dashboard/main_data.csv")

datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Filter data
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()


with st.sidebar:
    st.title("Habib Collection")
    try:
        start_date, end_date = st.date_input(
        label="Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
        )
    except ValueError:
        st.error("Masukkan tanggal mulai dan akhir yang valid.")
        start_date, end_date = min_date, max_date  

main_df = all_df[
    (all_df["order_purchase_timestamp"] >= str(start_date))
    & (all_df["order_purchase_timestamp"] <= str(end_date))
]

st.title("E-Commerce Data Analysis Dashboard")


rfm_df, customer_segment_df = create_rfm_df(main_df)

display_visualisasi_pertama(main_df)
display_visualisasi_kedua(main_df)
display_rfm(rfm_df)


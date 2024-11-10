import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


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


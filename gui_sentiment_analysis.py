!pip install geopy
!pip install folium streamlit-folium geopy

import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pickle
import os


os.getcwd()
df = pd.read_csv("df_full_hotel.csv")


# Hàm lấy lat, lon
def get_coordinates(address):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Cấu hình trang Streamlit
st.set_page_config(layout="wide")
# Display
st.title("Sentiment Analysis")
st.image('app.jpg', use_column_width=True)
st.write("HV: NGUYEN THI MY HUYEN")


st.session_state.df = df

# Kiểm tra xem 'selected_hotel' đã có trong session_state hay chưa
if 'selected_hotel' not in st.session_state:
    # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
    st.session_state.selected_hotel = None

# Chọn khách sạn từ dropbox
hotel_options = df['Hotel Name'].unique().tolist()
selected_hotel = st.selectbox("Chọn khách sạn",options=hotel_options,)
# Display the selected hotel
st.write("Bạn đã chọn:", selected_hotel)

# Cập nhật session_state dựa trên lựa chọn hiện tại
st.session_state.selected_hotel = selected_hotel

if st.session_state.selected_hotel:
    # Hiển thị thông tin khách sạn được chọn
    selected_hotel = df[df['Hotel Name'] == st.session_state.selected_hotel] 
    st.write("## THÔNG TIN VỀ KHÁCH SẠN")

    if not selected_hotel.empty:
        
        #ĐỊA CHỈ
        address = selected_hotel['Hotel Address'].values[0]
        st.write(" Địa chỉ: ",address)
                

        col1, col2, col3 = st.columns([1, 1, 1]) 
        
        with col1:
            st.write("RANK")
            hotel_rank = selected_hotel['Hotel Rank'].values[0]
            hotel_rank = ' '.join(hotel_rank.split()[:100])
            hotel_rank = hotel_rank.replace(" sao trên ","/")
            st.markdown(f"<span style='font-size: 70px;'>{hotel_rank}</span>", unsafe_allow_html=True)
            
        with col3:
            st.write("VỊ TRÍ")
            if address:
                lat, lon = get_coordinates(address)
                # if lat and lon:
                #     # Tạo bản đồ Folium
                #     m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")               
                #     # Thêm một điểm đánh dấu (marker) trên bản đồ
                #     folium.Marker([lat, lon], tooltip=address).add_to(m)               
                #     # Hiển thị bản đồ trong Streamlit
                #     st_folium(m, width=500, height=310)
                # else:
                #     st.write("Unable to find the location. Please try another address.")
                    
                if lat and lon:
                    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
                else:
                    st.write("Unable to find the location. Please try another address.")
        
        with col2:
            st.write("HÌNH ẢNH")
            if st.session_state.selected_hotel == "Khách sạn Mường Thanh Luxury Nha Trang (Muong Thanh Luxury Nha Trang Hotel)":
                file_name = "muongthanh.jpg"
            else: 
                file_name = "images.jpeg"
            st.image(file_name, use_column_width=True)
        
        #SCORE
        st.write("#### SCORE")

        col1, col2 = st.columns([1, 1]) 
        with col1:
            avg_score = selected_hotel['Total Score'].mean().round(1)
            st.markdown(f"**Số điểm đánh giá trung bình:** <span style='font-size: 100px;'>{avg_score}</span>", unsafe_allow_html=True)
        with col2:
            factor_columns = ['Vị trí', 'Độ sạch sẽ', 'Dịch vụ', 'Tiện nghi', 
                'Đáng giá tiền', 'Sự thoải mái và chất lượng phòng']
            factor_scores = selected_hotel[factor_columns].mean().round(1)
            # Tạo biểu đồ nằm ngang với matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            factor_scores.sort_values().plot(kind='barh', ax=ax)
            ax.set_xlabel('Điểm số', fontsize = 16)
            ax.set_ylabel('Yếu tố', fontsize = 16)
            ax.set_title('Điểm số các yếu tố', fontsize = 16)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = 16)
            # Thêm nhãn dữ liệu vào các cột
            for i, v in enumerate(factor_scores.sort_values()):
                ax.text(v + 0.05, i, str(v), color='black', va='center', fontsize=12)
            st.pyplot(fig)
        
        
        col1, col2 = st.columns([1, 1])   
        with col1: 
        # # # Xu hướng điểm số theo thời gian
            selected_hotel['Review Date'] = pd.to_datetime(selected_hotel['Review Date'])
            score_by_time = selected_hotel.groupby(selected_hotel['Review Date'].dt.to_period('M'))['Score'].mean()
            score_by_time.index = score_by_time.index.to_timestamp()  # Chuyển đổi chỉ mục về kiểu datetime

            # Vẽ biểu đồ đường với matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(score_by_time.index, score_by_time.values, marker='o', linestyle='-', color='b', label='Điểm số')

            # Thêm nhãn dữ liệu vào các điểm trên biểu đồ
            for i, (date, score) in enumerate(zip(score_by_time.index, score_by_time.values)):
                ax.text(date, score, f'{score:.1f}', fontsize=10, ha='right')

            # Định dạng trục x để hiển thị thời gian
            ax.set_xlabel('Thời gian', fontsize=12)
            ax.set_ylabel('Điểm số', fontsize=12)
            ax.set_title('Điểm số theo thời gian', fontsize=14)
            fig.autofmt_xdate(rotation=0)  

            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)
            
        with col2: 
            selected_hotel['Month'] = selected_hotel['Review Date'].dt.month
            # Tạo cột 'Month Year' để gộp dữ liệu theo tháng không phân biệt năm
            selected_hotel['Month Name'] = selected_hotel['Review Date'].dt.strftime('%B')  # Tên tháng

            # Nhóm dữ liệu theo tên tháng và tính điểm trung bình
            monthly_avg_score = selected_hotel.groupby('Month Name')['Score'].mean()

            # Đảm bảo thứ tự tháng là từ tháng 1 đến tháng 12
            monthly_avg_score = monthly_avg_score.reindex(
                ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
            )

            # Vẽ biểu đồ đường với matplotlib
            fig, ax = plt.subplots(figsize=(10,5.3))
            ax.plot(monthly_avg_score.index, monthly_avg_score.values, marker='o', linestyle='-', color='b')

            # Thêm nhãn dữ liệu vào các điểm trên biểu đồ
            for i, (month, score) in enumerate(zip(monthly_avg_score.index, monthly_avg_score.values)):
                ax.text(month, score, f'{score:.1f}', fontsize=10, ha='left')

            # Định dạng trục x để hiển thị tên tháng
            ax.set_xlabel('Tháng', fontsize=9)
            ax.set_ylabel('Điểm số Trung Bình', fontsize=10)
            ax.set_title('Điểm Trung Bình Theo Các Tháng Trong Năm', fontsize=12)
            ax.set_xticklabels(monthly_avg_score.index, rotation=0)  

            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)
                
            
        
        #SỐ LƯỢNG KHÁCH
        st.write("#### CUSTOMER PROFILE")    
        
        count_cus = selected_hotel['Reviewer ID'].nunique()
        st.markdown(f"**Số lượt khách hàng đã ở đây:** <span style='font-size: 100px;'>{count_cus}</span>", unsafe_allow_html=True)
        
        
        pivot_table = selected_hotel.pivot_table(values='Score', index='Room Type', columns='Group Name', aggfunc='mean')
        # Vẽ heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", linewidths=.5)
        plt.title('Average Score by Room Type and Customer Group')
        plt.xlabel('Group Name', fontsize=10)
        plt.ylabel('Room Type', fontsize=10)
        plt.xticks(fontsize=7, rotation=0)
        st.pyplot(plt)
        
        
        
        #COMMENT
        st.write("#### COMMENT")   
        col1, col2 = st.columns([1, 1]) 
        with col1:
            positive_comments = " ".join(comment for comment in selected_hotel['word_positive_list'].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)

            # Vẽ word cloud với matplotlib
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Ẩn trục để hiển thị tốt hơn
            plt.title('Positive Comments', fontsize=16)

            # Hiển thị word cloud trong Streamlit
            st.pyplot(plt)
        
        with col2:
            negative_comments = " ".join(comment for comment in selected_hotel['word_negative_list'].dropna())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)

            # Vẽ word cloud với matplotlib
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Ẩn trục để hiển thị tốt hơn
            plt.title('Negative Comments', fontsize=16)

            # Hiển thị word cloud trong Streamlit
            st.pyplot(plt)
        
    else:
        st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel}")


import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import pickle
import os


# Cấu hình trang Streamlit
st.set_page_config(layout="wide")

# os.getcwd()
df = pd.read_csv("df_full_hotel.csv")

# Các hàm xử lý tiếng việt -- do pipeline file pkl > 25mb ko up lên git đc
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

##LOAD EMOJICON
file = open('emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()


def process_text(text, emoji_dict, english_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT ENGLISH
        sentence = ' '.join(english_dict[word] if word in english_dict else word for word in sentence.split())
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + " . "
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document

# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def process_special_word(text):
    # có thể có nhiều từ đặc biệt cần ráp lại với nhau
    new_text = ''
    text_lst = text.split()
    i= 0
    # không, chẳng, chả...
    if 'không' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không' or word == 'ko' or word == 'hong' or word == 'không hề' or word == 'ko hề' or word == 'hong hề':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

import re
# Hàm để chuẩn hóa các từ có ký tự lặp
def normalize_repeated_characters(text):
    # Thay thế mọi ký tự lặp liên tiếp bằng một ký tự đó
    # Ví dụ: "lònggggg" thành "lòng", "thiệtttt" thành "thiệt"
    return re.sub(r'(.)\1+', r'\1', text)
    
def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document


def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        word_lower = word.lower()
        count = document_lower.count(word_lower)  # Đếm số lần từ xuất hiện trong danh sách các từ
        if count > 0:
            word_count += count
            word_list.append(word)

    return word_count, word_list


def pre_process(comment,emoji_dict, english_dict, teen_dict, wrong_lst,stopwords_lst):
  comment = process_text(comment, emoji_dict, english_dict, teen_dict, wrong_lst)
  print(comment)
  comment = covert_unicode(comment)
  print(comment)
  comment = covert_unicode(comment)
  print(comment)
  comment = process_special_word(comment)
  print(comment)
  comment = normalize_repeated_characters(comment)
  print(comment)
  comment = process_postag_thesea(comment)
  print(comment)
  comment = remove_stopword(comment, stopwords_lst)
  print(comment)
  return comment

# Đọc vectorizer
with open("vectorizer.pkl", 'rb') as file:  
    vectorizer = pickle.load(file)


# Đọc model
with open("sa_model.pkl", 'rb') as file:  
    sa_model = pickle.load(file)


# # Hàm lấy lat, lon
# def get_coordinates(address):
#     geolocator = Nominatim(user_agent="geoapiExercises")
#     location = geolocator.geocode(address)
#     if location:
#         return location.latitude, location.longitude
#     else:
#         return None, None

# df[['lat','lon']] = df['Hotel Address'].apply(lambda x: get_coordinates(x))

menu = ["Home", "Business Analysis","Recommendation"]
choice = st.sidebar.selectbox('Menu', menu)


if choice == 'Home':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("HV: NGUYEN THI MY HUYEN")
    st.image('app.jpg', use_column_width=True)
elif choice == "Business Analysis":
    # Display
    st.image('app.jpg', use_column_width=True)
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
                    

            col1, col2 = st.columns([1, 1]) 
            
            with col1:
                st.write("RANK")
                hotel_rank = selected_hotel['Hotel Rank'].values[0]
                hotel_rank = ' '.join(hotel_rank.split()[:100])
                hotel_rank = hotel_rank.replace(" sao trên ","/")
                st.markdown(f"<span style='font-size: 100px;'>{hotel_rank}</span>", unsafe_allow_html=True)
                
            # with col3:
            #     st.write("VỊ TRÍ")
            #     if address:
            #         lat = selected_hotel['lat'].values[0]
            #         lon = selected_hotel['lon'].values[0]
            #         if lat and lon:
            #             st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
            #         else:
            #             st.write("Unable to find the location. Please try another address.")
            
            with col2:
                st.write("HÌNH ẢNH")
                file_name = "muongthanh.jpg"
                if st.session_state.selected_hotel != "Khách sạn Mường Thanh Luxury Nha Trang (Muong Thanh Luxury Nha Trang Hotel)":
                    file_name = "images.jpeg"
                st.image(file_name, use_column_width=True)
            
            #SCORE
            st.write("### SCORE")

            col1, col2 = st.columns([1, 1]) 
            with col1:
                avg_score = selected_hotel['Total Score'].mean().round(1)
                st.markdown(f"**Số điểm đánh giá trung bình:** <span style='font-size: 50px;'>{avg_score}</span>", unsafe_allow_html=True)
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
                selected_hotel['Review Date'] = pd.to_datetime(selected_hotel['Review Date'], errors='coerce')
                selected_hotel['Month'] = selected_hotel['Review Date'].dt.month
                score_by_time = selected_hotel.groupby(selected_hotel['Month'])['Score'].mean()

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
            st.write("### CUSTOMER")    
            
            count_cus = selected_hotel['Reviewer ID'].nunique()
            st.markdown(f"**Số lượt khách hàng đã ở đây:** <span style='font-size: 50px;'>{count_cus}</span>", unsafe_allow_html=True)
            
            
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
            st.write("### COMMENT")  
            # Vẽ biểu đồ tròn
            col1, col2, col3 = st.columns([1, 1, 1]) 
            with col1:
                fig, ax = plt.subplots(figsize=(6,6))
                # Tạo pivot table để đếm số lượng Reviewer ID theo từng nhãn 'label'
                l_count = selected_hotel.pivot_table(values='Reviewer ID', index='label', aggfunc='nunique')
                
                # Tạo từ điển để ánh xạ các giá trị label
                label_mapping = {-1: "Negative", 0: "Neutral", 1: "Positive"}
                # Áp dụng ánh xạ để thay đổi nhãn
                labels = [label_mapping[label] for label in l_count.index]

                # Vẽ biểu đồ tròn
                ax.pie(l_count['Reviewer ID'], labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Đảm bảo biểu đồ tròn không bị méo
                st.pyplot(fig)

            col1, col2 = st.columns([1, 1]) 
            with col1:
                p_count = selected_hotel[selected_hotel['label']==1]['Reviewer ID'].nunique()
                st.markdown(f"**Number of Positive Comment:** <span style='font-size: 50px;'>{p_count}</span>", unsafe_allow_html=True)
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
                n_count = selected_hotel[selected_hotel['label']==-1]['Reviewer ID'].nunique()
                st.markdown(f"**Number of Negative Comment:** <span style='font-size: 50px;'>{n_count}</span>", unsafe_allow_html=True)
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

elif choice == "Recommendation":
    
    flag = False
    lines = None
    content = st.text_area(label="Input your content:")
    if content!="":
        lines = np.array([content])
        flag = True
    
    if flag:
        st.write("Bên dưới là prediction cho comment: ")
        if len(lines)>0:
            st.code(lines)
            x_new = pre_process(content,emoji_dict, english_dict, teen_dict, wrong_lst, stopwords_lst)
            st.write(x_new)
            x_new = vectorizer.transform([x_new])
            y_pred_new = sa_model.predict(x_new)
        if y_pred_new == 1:
            st.write("POSITIVE COMMENT")
        elif y_pred_new == -1:
            st.write("NEGATIVE COMMENT")
        else:
            st.write("NEUTRAL COMMENT")
   
            


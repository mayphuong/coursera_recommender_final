import streamlit as st
import pandas as pd

st.title("Trung Tam Tin Hoc")
st.subheader("How to run streamlit app")

menu = ["Home", "About", 'Sử dụng các điều khiển'
        , 'Gợi ý điều khiển project 1'
        , 'Gợi ý điều khiển project 2'
        , 'Gợi ý điều khiển project 3']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    st.subheader("Streamlit From MacOS")
elif choice == 'About':
    st.subheader("[Trung Tam Tin Hoc](https://csc.edu.vn)")
elif choice == 'Sử dụng các điều khiển':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)

    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")

    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")

    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)

    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)

    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)

    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)

    # 8. Date Input
    st.subheader("8. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)

    # 9. Time Input
    st.subheader("9. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)

    # 10. Display JSON
    st.subheader("10. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)

    # 11. Display Raw Code
    st.subheader("11. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)

elif choice == 'Gợi ý điều khiển project 1':
    st.subheader("Gợi ý điều khiển project 1: Sentiment Analysis")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file

    type = st.radio("Chọn cách nhập liệu", options=["Nhập liệu vào text area", "Nhập nhiều dòng dữ liệu trực tiếp", "Upload file"])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Nhập liệu vào text area":
        st.subheader("Nhập dữ liệu vào text area")
        content = st.text_area("Nhập ý kiến:")
        # Nếu người dùng nhập dữ liệu đưa content này vào thành 1 dòng trong DataFrame
        if content:
            df = pd.DataFrame([content], columns=["Ý kiến"])
            st.dataframe(df)
            st.table(df)
    
    # Nếu người dùng chọn nhập nhiều dòng dữ liệu trực tiếp vào một table
    elif type == "Nhập nhiều dòng dữ liệu trực tiếp":
        st.subheader("Nhập nhiều dòng dữ liệu trực tiếp")        
        df = pd.DataFrame(columns=["Ý kiến"])
        for i in range(5):
            # df = df.append({"Ý kiến": st.text_area(f"Nhập ý kiến {i+1}:")}, ignore_index=True)
            row = {"Ý kiến": st.text_area(f"Nhập ý kiến {i+1}:")}
            df = df.append(row, ignore_index=True)
        st.dataframe(df) 
    
    # Nếu người dùng chọn upload file
    elif type == "Upload file":
        st.subheader("Upload file")
        # Upload file
        uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "txt"])
        if uploaded_file is not None:
            # Đọc file dữ liệu
            df = pd.read_csv(uploaded_file)
            st.write(df)
    # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau

elif choice == 'Gợi ý điều khiển project 2':
    # Tạo 1 dictionary gồm có 10 phần tử
    # Với key là các mã số khách hàng ngẫu nhiên gồm 8 ký tự trong đó có 4 ký tự chữ và 4 ký tự chuỗi
    # Và value là họ tên tương ứng
    data = {
        'KH001A': 'Nguyễn Văn A',
        'KH002B': 'Trần Thị B',
        'KH003C': 'Phạm Văn C',
        'KH004D': 'Lê Thị D',
        'KH005E': 'Hoàng Văn E',
        'KH006F': 'Nguyễn Thị F',
        'KH007G': 'Trần Văn G',
        'KH008H': 'Phạm Thị H',
        'KH009I': 'Lê Văn I',
        'KH010K': 'Hoàng Thị K'
    }
    # Tạo DataFrame từ dictionary
    df_KH = pd.DataFrame(list(data.items()), columns=['Mã số', 'Họ tên'])
    
    # In df_KH ra màn hình dạng table
    st.write("Danh sách khách hàng:")
    st.write(df_KH)
    
    # lấy ngẫu nhiên 3 khách hàng từ df_KH
    df_sample = df_KH.sample(3)

    # In 3 khách hàng này ra màn hình
    st.write("Danh sách khách hàng ngẫu nhiên:")
    st.write(df_sample)
    st.write("##### Đề xuất sản phẩm cho khách hàng dựa trên Mã số khách hàng")
    
    # Tạo một điều khiển và đưa khách hàng ngẫu nhiên này vào đó
    st.write("Chọn khách hàng:")
    selected = st.selectbox("Chọn khách hàng", df_sample['Họ tên'])
    st.write("Khách hàng đã chọn:", selected)
    
    # Từ khách hàng được chọn này lấy mã số tương ứng và tiếp tục xử lý phần đề xuất sản phẩm cho khách hàng này
    st.write("Xử lý và hiển thị thông tin Đề xuất sản phẩm cho khách hàng...")
    st.write("##### Đề xuất sản phẩm cho khách hàng dựa trên sản phẩm")
    
    # Tạo dataframe danh sách hàng hóa gồm 10 sản phẩm gồm mã sản phẩm, tên sản phẩm, mô tả tóm tắt khoảng 50 ký tự và giá bán
    data = {
        'Mã SP': ['SP001', 'SP002', 'SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP008', 'SP009', 'SP010'],
        'Tên SP': ['Áo thun nam', 'Áo sơ mi nữ', 'Quần jean nam', 'Quần legging nữ', 'Áo khoác nam', 'Áo len nữ', 'Quần kaki nam', 'Quần legging nữ', 'Áo khoác nam', 'Áo len nữ'],
        'Mô tả': ['Áo thun nam hàng hiệu', 'Áo sơ mi nữ hàng hiệu', 'Quần jean nam hàng hiệu', 'Quần legging nữ hàng hiệu', 'Áo khoác nam hàng hiệu', 'Áo len nữ hàng hiệu', 'Quần kaki nam hàng hiệu', 'Quần legging nữ hàng hiệu', 'Áo khoác nam hàng hiệu', 'Áo len nữ hàng hiệu'],
        'Giá': [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    }
    df_SP = pd.DataFrame(data)
    
    # In danh sách sản phẩm ra table
    st.dataframe(df_SP)
    
    # Tạo điều khiển để người dùng chọn sản phẩm
    st.write("##### 1. Chọn sản phẩm")
    selected_SP = st.selectbox("Chọn sản phẩm", df_SP['Tên SP'])
    st.write("Sản phẩm đã chọn:", selected_SP)

    # Tìm sản phẩm liên quan đến sản phẩm đã chọn
    st.write("##### 2. Sản phẩm liên quan")
    
    # Lấy thông tin sản phẩm đã chọn
    SP = df_SP[df_SP['Tên SP'] == selected_SP]
    mo_ta_chon = SP['Mô tả'].iloc[0].lower()
    
    # Gợi ý sản phẩm liên quan dựa theo mô tả của sản phẩm đã chọn, chuyển thành chữ thường trước khi tìm kiếm 
    related_SP = df_SP[df_SP['Mô tả'].str.lower().str.contains(mo_ta_chon, na=False)]
    
    # In danh sách sản phẩm liên quan ra màn hình
    st.write("Danh sách sản phẩm liên quan:")
    st.dataframe(related_SP)

    # Từ sản phẩm đã chọn này, người dùng có thể xem thông tin chi tiết của sản phẩm, xem hình ảnh sản phẩm
    # hoặc thực hiện các xử lý khác
    # tạo điều khiển để người dùng tìm kiếm sản phẩm dựa trên thông tin người dùng nhập
    st.write("##### 3. Tìm kiếm sản phẩm")
    search = st.text_input("Nhập thông tin tìm kiếm")
    
    # Tìm kiếm sản phẩm dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    result = df_SP[df_SP['Mô tả'].str.lower().str.contains(search.lower())]    
    
    # In danh sách sản phẩm tìm được ra màn hình
    st.write("Danh sách sản phẩm tìm được:")
    st.dataframe(result)
    # Từ danh sách sản phẩm tìm được này, người dùng có thể xem thông tin chi tiết của sản phẩm, xem hình ảnh sản phẩm...

elif choice=='Gợi ý điều khiển project 3':
    st.write("##### 1. Some data")
    # Chọn nhập mã khách hàng hoặc nhập thông tin khách hàng vào dataframe
    st.write("##### 1. Chọn cách nhập thông tin khách hàng")
    type = st.radio("Chọn cách nhập thông tin khách hàng", options=["Nhập mã khách hàng", "Nhập thông tin khách hàng vào dataframe"])
    if type == "Nhập mã khách hàng":
        # Nếu người dùng chọn nhập mã khách hàng
        st.subheader("Nhập mã khách hàng")
        
        # Tạo điều khiển để người dùng nhập mã khách hàng
        customer_id = st.text_input("Nhập mã khách hàng")
        
        # Nếu người dùng nhập mã khách hàng, thực hiện các xử lý tiếp theo
        # Đề xuất khách hàng thuộc cụm nào
        # In kết quả ra màn hình
        st.write("Mã khách hàng:", customer_id)
        st.write("Phân cụm khách hàng...")

    else:
        # Nếu người dùng chọn nhập thông tin khách hàng vào dataframe có 3 cột là Recency, Frequency, Monetary
        st.write("##### 2. Thông tin khách hàng")
        
        # Tạo điều khiển table để người dùng nhập thông tin khách hàng trực tiếp trên table
        st.write("Nhập thông tin khách hàng")
        
        # Tạo dataframe để người dùng nhập thông tin khách hàng
        df_customer = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])

        for i in range(5):
            st.write(f"Khách hàng {i+1}")
            # Tạo các slider để nhập giá trị cho cột Recency, Frequency, Monetary
            recency = st.slider("Recency", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary", 1, 1000, 100, key=f"monetary_{i}")
            
            # Cũng có thể thay bằng các điều khiển khác như number_input...
            # Thêm thông tin khách hàng vừa nhập vào dataframe
            df_customer = df_customer.append({"Recency": recency, "Frequency": frequency, "Monetary": monetary}, ignore_index=True)            
        # Thực hiện phân cụm khách hàng dựa trên giá trị của 3 cột này

        # In kết quả ra màn hình
        st.write("##### 3. Phân cụm khách hàng")
        st.write(df_customer)
        st.write("Phân cụm khách hàng...")
        # Từ kết quả phân cụm khách hàng, người dùng có thể xem thông tin chi tiết của từng cụm khách hàng, xem biểu đồ, thống kê...
        # hoặc thực hiện các xử lý khác
# Done
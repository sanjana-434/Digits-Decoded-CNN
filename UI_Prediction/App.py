import streamlit as st
st.set_page_config(page_title='digit recognition', layout="wide")

st.title('MNIST Classification using CNN')
st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.image('https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp')

with col2:
    st.markdown('MNIST digits dataset is a vast collection of handwritten digits.')
    st.markdown('This dataset is used for training image processing systems.')
    st.markdown('It is also a very popular dataset used in universities to explain the concepts of classification in machine learning. ')
    st.markdown('It contains 60,000 training images and 10,000 testing images.')


st.sidebar.write('Developed by ')
st.sidebar.write('Sanjana R - 21PD31')
from paperTrans import paperTrans
import streamlit as st
import pandocfilters as pf
import shutil
import zipfile
import os
import random

st.columns([1,6,1])[1].image('imgs/paperTrans.svg')
st.title("paperTrans")
st.subheader("Change paper's formatting and translate to Markdown.")

file_id = 'file'+str(0)
while file_id in os.listdir():
    file_id = 'file'+str(random.randint(1,100))

col_left, col_right = st.columns(2)
lang = col_right.selectbox("Traget language", ["zh-tw","en"])
form = col_left.radio("Output form", ["Markdown"])
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:

    file = file_id+'.pdf'
    with open(file, 'wb') as f:
        f.write(uploaded_file.read())

    # Run paperTrans
    with st.spinner('Processing PDF:'):
        paperTrans(file, lang)

        if form == 'Markdown':
            # Zip the folder
            with zipfile.ZipFile(file_id+'.zip', 'w', zipfile.ZIP_STORED) as zip_obj:
                for folder_name, subfolders, filenames in os.walk(file_id):
                    for filename in filenames:
                        file_path = os.path.join(folder_name, filename)
                        zip_obj.write(file_path)

            # Download the zip file
            data=open(file_id+'.zip', 'rb')
            st.download_button("Download file", file_name=file_id+'.zip', data=data)

        try:
            # Delete the uploaded file
            os.remove(file)

            # Delete the generated file
            if form == 'Markdown':
                os.remove(file_id+'.zip')
            elif form == 'PDF':
                os.remove(file_id+'.pdf')
            os.remove('image_.png')

            # Delete the "file" folder
            shutil.rmtree(file_id)
        except:
            pass





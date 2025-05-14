import streamlit as st
import pandas as pd

import mylib as ml



# ... 现有代码 ...
def main():
    st.title('Calculate Metrics')
    st.write('This app calculates the metrics for a given dataset')
    #显示图片
    st.image('image.png', caption='upload file format', width=700)
    #upload a csv file
    st.write('Upload a CSV file,there are at least two columns in the file. Row 1 are column names. The first column is the actual labels and the other columns are the predicted values')
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        st.write('The number of rows and columns are:')
        st.write(df.shape)
    # calculate metrics
        st.write('The metrics are:')
        methods = df.columns[1:]
        results = {}
        for method in methods:
            # 处理NaN值
            if df[method].isnull().any() or df.iloc[:, 0].isnull().any():
                temp = df.dropna(subset=[method, df.columns[0]])
                results[method] = ml.scores(temp.iloc[:, 0], temp[method])
            else:
                results[method] = ml.scores(df.iloc[:, 0], df[method])
        st.write(results)


    # ... existing code ...

    # 或者可以从输入框中接收数据，输入框为多列数据
    st.write('Or you can input the data from the input box (use TAB to separate columns)')
    default_data = '''actual\tpred1\tpred2
    1\t1\t0
    0\t0\t1
    1\t1\t1
    0\t0\t0
    1\t1\t0
    0\t0\t1
    1\t1\t1
    0\t0\t0'''
    data = st.text_area('Input the data (TAB-separated)', value=default_data, height=200)

    if data:
        try:
            from io import StringIO
            df_input = pd.read_csv(StringIO(data), sep='\t')  # 使用\t作为分隔符
            st.write('Input data preview:')
            st.write(df_input.shape)
            st.write(df_input.head())
            
            # 计算指标
            if len(df_input.columns) >= 2:
                methods = df_input.columns[1:]
                results = {}
                for method in methods:
                    #当前method列如果有缺失值，则去除缺失值
                    if df_input[method].isnull().any():
                        temp = df_input.dropna(subset=[method])
                        results[method] = ml.scores(temp.iloc[:, 0], temp[method])
                    else:
                        results[method] = ml.scores(df_input.iloc[:, 0], df_input[method])
                st.write('Metrics results:')
                st.write(pd.DataFrame(results,index='Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR,tp,fn,tn,fp'.split(',')))
            else:
                st.error('Input data must contain at least 2 columns (actual and predicted values)')
        except Exception as e:
            st.error(f'Error processing input data: {str(e)}')

    # ... existing code ...
    # # 在EPEL预计算分数的vcf文件中匹配test1_data.txt中的突变
    # match_columns.match(file_a=r'E:\WLH\bioinformatics\突变资料\synMall\叶晨-同义突变数据库\Annotator\user_output\maeDSM_indep1_20240811_22_09\test1_mutations.csv',
    #             file_db=r'E:\WLH\bioinformatics\bio_codes\driver-syns-Bichuanmei\EPEL_score.zip',
    #             matched_file=r'E:\WLH\bioinformatics\突变资料\synMall\叶晨-同义突变数据库\Annotator\user_output\maeDSM_indep1_20240811_22_09\test1_198_198_EPEL.csv',
    #             header_a=0,
    #             header_db=0
    #             )
    #在EPEL预计算分数的vcf文件中匹配test2中的突变
    # match_columns.match(file_a=r'E:\WLH\bioinformatics\突变资料\synMall\叶晨-同义突变数据库\Annotator\user_output\maeDSM_indep2_20240812_22_19\test2_mutations.csv',
    #             file_db=r'E:\WLH\bioinformatics\bio_codes\driver-syns-Bichuanmei\EPEL_score.zip',
    #             matched_file=r'E:\WLH\bioinformatics\突变资料\synMall\叶晨-同义突变数据库\Annotator\user_output\maeDSM_indep2_20240812_22_19\test2_96_96_EPEL.csv',
    #             header_a=0,
    #             header_db=0
    #             )

if __name__ == "__main__":
    main()
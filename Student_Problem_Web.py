from ctypes.wintypes import tagMSG
import pandas as pd
#streamlit==1.23.1
import streamlit as st
import pickle

st.title('問題行動予測アプリ')

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください。",type='csv')

#文字列の列を抜き出す（d_marge01_A～d_marge36_A,それぞれ末尾D,F,H）
str_col_name = []
alphabetlist = ("A","D","F","H")
for alphabet in alphabetlist:
    for num in range(36):
        if(num < 9):
            str_col_name.append("d_marge0" + str(num +1)  + "_" + alphabet)
        else:
            str_col_name.append("d_marge" + str(num +1)  + "_" + alphabet )

#予測に使用しないカラムを取得
drop_alphabetlist = ["a","b","c","d"]
drop_col_name = []
for dropalphabet in drop_alphabetlist:
    drop_col_name.append(dropalphabet +"_id")
    drop_col_name.append(dropalphabet +"_StudentId")
    drop_col_name.append(dropalphabet +"_SchoolNumber")
    drop_col_name.append(dropalphabet +"_SchoolGrade")
    drop_col_name.append(dropalphabet +"_SchoolClass")
    drop_col_name.append(dropalphabet +"_PersonalNumber")
    drop_col_name.append(dropalphabet +"_MesurementDate")
    drop_col_name.append(dropalphabet +"_replacement_1")
    drop_col_name.append(dropalphabet +"_replacement_2")
    if(dropalphabet == "a"):
        drop_col_name.append(dropalphabet +"_CreateDate")
    else:
        drop_col_name.append(dropalphabet +"_CreatedDate")

#文字列のカラムを取得
stringlist = ['Ali','Arc','Ari','Dlc','Dli','Drc','Dri','Flc','Fli','Frc','Fri','Hlc','Hli','Hrc','Hri']
categories_name = []
for alphabet in alphabetlist:
    for num in range(36):
        if(num < 9):
            for string in stringlist:
                categories_name.append("d_marge0" + str(num +1)  + "_" + alphabet + "_" + string)
                #print("d_marge" + str(num +1) + "_" + alphabet)
        else:
            for string in stringlist:
                categories_name.append("d_marge" + str(num +1)  + "_" + alphabet+ "_" + string )
            #print("d_marge" + str(num +1) + "_" + alphabet)

#モデルを読み込む
with open('Student_Problem.pkl','rb') as f:
    model2= pickle.load(f)

#データの読み込み
if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        engine='python',
        na_values='-',
        header=0)
    
    chackcolname = [str_col_name,drop_col_name]
    for checkcol in chackcolname:
        # str_col_name リストと df データフレームのカラム名の一致をチェック
        invalid_cols = [col for col in checkcol if col not in df.columns]

        if len(invalid_cols) > 0:
            # エラーメッセージを表示してプログラムを終了
            st.error(f"次のカラム名が存在しません。ファイルを確認してください: {', '.join(invalid_cols)}")
            # 以降の処理をスキップ
            st.stop()
      
    str_df = df[str_col_name]
    #ダミー変数変化用の列を追加
    for column_name in categories_name:
        df[column_name] = 0  

    for alphabet in alphabetlist:
        for num in range(36):
            if num < 9:
                prefix = 'd_marge0' + str(num + 1) + '_' + alphabet + '_'
            else:
                prefix = 'd_marge' + str(num + 1) + '_' + alphabet + '_'
            for string in stringlist:
                column_name = prefix + string
                df[column_name] = df[prefix[:-1]].apply(lambda x: 1 if x == string else 0)

    #もとのデータフレームから文字列の列だけ削除
    df2 = df.drop(str_col_name,axis = 1)

    #欠損値を0で穴埋め
    df2 = df2.fillna(0)
    #モデル学習に使用しないカラムを削除
    df_x = df2.drop(drop_col_name,axis=1)

    

    try:
        x_pred = model2.predict(df_x)
    except Exception as e:
        st.error(f"予測時にエラーが発生しました。ファイルを確認してください。: {e}")
        st.stop()

    option = st.selectbox(
    '表示選択',
    ['すべて表示','問題行動を起こす可能性が高い児童のみ表示','問題行動を起こす可能性が低い児童のみ表示']
    )

    df_x_pred = pd.DataFrame(x_pred, columns=['Predict'])
    df_x_pred['問題行動を起こす可能性'] = ["可能性は低いです。" if p == 0 else"問題行動を起こす可能性があります。" for p in df_x_pred['Predict']]
    st.markdown('### 予測結果')
    
    t = df['a_StudentId'] 

    if option == 'すべて表示':
        df_x_pred = df_x_pred.drop(['Predict'],axis=1)

        df_y = pd.concat([t,df_x_pred],axis = 1)
        df_y_new = df_y.rename(columns = {'a_StudentId': 'StudentId'})
    
        st.write(df_y_new)

    elif option == '問題行動を起こす可能性が高い児童のみ表示':
        df_y = pd.concat([t,df_x_pred],axis = 1)
        #予測結果が0(可能性低)のインデックスを取得
        indexNames = df_y[ (df_y['Predict'] == 0)].index
        #その行を削除
        df_y.drop(indexNames , inplace=True)
        #0,1の行を削除
        df_y = df_y.drop(['Predict'],axis=1)
        
        df_y_new = df_y.rename(columns = {'a_StudentId': 'StudentId'})
        st.write(df_y_new)
    else:   
        df_y = pd.concat([t,df_x_pred],axis = 1)
        #予測結果が0(可能性低)のインデックスを取得
        indexNames = df_y[ (df_y['Predict'] == 1)].index
        #その行を削除
        df_y.drop(indexNames , inplace=True)
        #0,1の行を削除
        df_y = df_y.drop(['Predict'],axis=1)
        
        df_y_new = df_y.rename(columns = {'a_StudentId': 'StudentId'})
        st.write(df_y_new)


    





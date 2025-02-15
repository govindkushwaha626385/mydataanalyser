# Create an Interactive Data Analytics Portal with Streamlit in 7 Steps
# import libraries
import pandas as pd 
import plotly.express as px
import streamlit as st
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title='Consoleflare Analytics Portal',
    page_icon='📊'
)
#title
st.title(':rainbow[Data Analytics Portal]')
st.subheader(':gray[Explore Data with ease.]',divider='rainbow')

file = st.file_uploader('Drop csv or excel file',type=['csv','xlsx'])
if(file!=None):
    if(file.name.endswith('csv')):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)   
    
    st.dataframe(data)
    st.info('File is successfully Uploaded',icon='🚨')

    st.subheader(':rainbow[Basic information of the dataset]',divider='rainbow')
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['Summary','Info', 'Top and Bottom Rows','Data Types','Columns', 'Missing Values'])

    with tab1:
        st.write(f'There are {data.shape[0]} rows in dataset and  {data.shape[1]} columns in the dataset')
        st.subheader(':gray[Statistical summary of the dataset]')
        st.dataframe(data.describe())
    # with tab2:
        # st.dataframe(data.info()) 
    with tab3:
        st.subheader(':gray[Top Rows]')
        toprows = st.slider('Number of rows you want',1,data.shape[0],key='topslider')
        st.dataframe(data.head(toprows))
        st.subheader(':gray[Bottom Rows]')
        bottomrows = st.slider('Number of rows you want',1,data.shape[0],key='bottomslider')
        st.dataframe(data.tail(bottomrows))
    with tab4:
        st.subheader(':grey[Data types of column]')
        st.dataframe(data.dtypes)
    with tab5:
        st.subheader('Column Names in Dataset')
        st.write(list(data.columns))
    with tab6:
         st.subheader("Missing Values count in Each Column")
         st.dataframe(data.isna().sum())    
    
    st.subheader(':rainbow[Column Values To Count]',divider='rainbow')
    with st.expander('Value Count'):
        col1,col2 = st.columns(2)
        with col1:
          column  = st.selectbox('Choose Column name',options=list(data.columns))
        with col2:
            toprows = st.number_input('Top rows',min_value=1,step=1)
        
        count = st.button('Count')
        if(count==True):
            result = data[column].value_counts().reset_index().head(toprows)
            st.dataframe(result)
            st.subheader('Visualization',divider='gray')
            fig = px.bar(data_frame=result,x=column,y='count',text='count',template='plotly_white')
            st.plotly_chart(fig)
            fig = px.line(data_frame=result,x=column,y='count',text='count',template='plotly_white')
            st.plotly_chart(fig)
            fig = px.pie(data_frame=result,names=column,values='count')
            st.plotly_chart(fig)


    st.subheader(':rainbow[Groupby : Simplify your data analysis]',divider='rainbow')
    st.write('The groupby lets you summarize data by specific categories and groups')
    with st.expander('Group By your columns'):
        col1,col2,col3 = st.columns(3)
        with col1:
            groupby_cols = st.multiselect('Choose your column to groupby',options = list(data.columns))
        with col2:
            operation_col = st.selectbox('Choose column for operation',options=list(data.columns))
        with col3:
            operation = st.selectbox('Choose operation',options=['sum','max','min','mean','median','count'])
        
        if(groupby_cols):
            result = data.groupby(groupby_cols).agg(
                newcol = (operation_col,operation)
            ).reset_index()

            st.dataframe(result)

            st.subheader(':gray[Data Visualization]',divider='gray')
            graphs = st.selectbox('Choose your graphs',options=['line','bar','scatter','pie','sunburst'])
            if(graphs=='line'):
                x_axis = st.selectbox('Choose X axis',options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(result.columns))
                color = st.selectbox('Color Information',options= [None] +list(result.columns))
                fig = px.line(data_frame=result,x=x_axis,y=y_axis,color=color,markers='o')
                st.plotly_chart(fig)
            elif(graphs=='bar'):
                 x_axis = st.selectbox('Choose X axis',options=list(result.columns))
                 y_axis = st.selectbox('Choose Y axis',options=list(result.columns))
                 color = st.selectbox('Color Information',options= [None] +list(result.columns))
                 facet_col = st.selectbox('Column Information',options=[None] +list(result.columns))
                 fig = px.bar(data_frame=result,x=x_axis,y=y_axis,color=color,facet_col=facet_col,barmode='group')
                 st.plotly_chart(fig)
            elif(graphs=='scatter'):
                x_axis = st.selectbox('Choose X axis',options=list(result.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(result.columns))
                color = st.selectbox('Color Information',options= [None] +list(result.columns))
                size = st.selectbox('Size Column',options=[None] + list(result.columns))
                fig = px.scatter(data_frame=result,x=x_axis,y=y_axis,color=color,size=size)
                st.plotly_chart(fig)
            elif(graphs=='pie'):
                values = st.selectbox('Choose Numerical Values',options=list(result.columns))
                names = st.selectbox('Choose labels',options=list(result.columns))
                fig = px.pie(data_frame=result,values=values,names=names)
                st.plotly_chart(fig)
            elif(graphs=='sunburst'):
                path = st.multiselect('Choose your Path',options=list(result.columns))
                fig = px.sunburst(data_frame=result,path=path,values='newcol')
                st.plotly_chart(fig)
    

    st.subheader(':rainbow[Data Visualization using Graphs]',divider='gray')
    graphs = st.selectbox('Choose your graphs',options=['line','bar','scatter','pie','sunburst', 'box', 'hist'])
    if(graphs=='line'):
                x_axis = st.selectbox('Choose X axis',options=list(data.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(data.columns))
                color = st.selectbox('Color Information',options= [None] +list(data.columns))
                fig = px.line(data_frame=data,x=x_axis,y=y_axis,color=color,markers='o')
                st.plotly_chart(fig)
    elif(graphs=='bar'):
                 x_axis = st.selectbox('Choose X axis',options=list(data.columns))
                 y_axis = st.selectbox('Choose Y axis',options=list(data.columns))
                 color = st.selectbox('Color Information',options= [None] +list(data.columns))
                 facet_col = st.selectbox('Column Information',options=[None] +list(data.columns))
                 fig = px.bar(data_frame=data,x=x_axis,y=y_axis,color=color,facet_col=facet_col,barmode='group')
                 st.plotly_chart(fig)
    elif(graphs=='scatter'):
                x_axis = st.selectbox('Choose X axis',options=list(data.columns))
                y_axis = st.selectbox('Choose Y axis',options=list(data.columns))
                color = st.selectbox('Color Information',options= [None] +list(data.columns))
                size = st.selectbox('Size Column',options=[None] + list(data.columns))
                fig = px.scatter(data_frame=data,x=x_axis,y=y_axis,color=color,size=size)
                st.plotly_chart(fig)
    elif(graphs=='pie'):
                values = st.selectbox('Choose Numerical Values',options=list(data.columns))
                names = st.selectbox('Choose labels',options=list(data.columns))
                fig = px.pie(data_frame=data,values=values,names=names)
                st.plotly_chart(fig)
    elif(graphs=='sunburst'):
                path = st.multiselect('Choose your Path',options=list(data.columns))
                fig = px.sunburst(data_frame=data,path=path,values='newcol')
                st.plotly_chart(fig)
    elif(graphs == 'hist'):
                column = st.selectbox('Choose Column ', options = list(data.select_dtypes(include=['int64', 'float64']).columns))
                hue = st.selectbox('Choose Column ', options = list(data.select_dtypes(include=['object']).columns))

                # Create Histogram with Hue using Plotly Express
                fig = px.histogram(data, x=column, color=hue, nbins=30,
                   histnorm='probability density', opacity=0.6, title="Histogram with KDE & Hue")

                # Compute and Overlay KDE for Each Category
                for category in data[hue].unique():
                    subset = data[data[hue] == category][column]  # Filter by category
                    kde = sns.kdeplot(subset, bw_adjust=1, fill=False)

                    # Extract KDE Data
                    x_kde = kde.get_lines()[-1].get_xdata()
                    y_kde = kde.get_lines()[-1].get_ydata()

                    # Add KDE Curve for Each Category
                    fig.add_trace(go.Scatter(x=x_kde, y=y_kde, mode="lines", name=f"KDE - {category}"))

                # Display in Streamlit
                st.plotly_chart(fig)

    elif (graphs == 'box'):
                column = st.selectbox('Choose Column ', options = list(data.columns))
                fig = px.box(data, y= column)
                st.plotly_chart(fig)    

                if data[column].dtype in ['int64', 'float64']:
                    st.write(f"Column '{column}' is numeric")
            
                    remove_outlier = st.button("Remove outliers of selected column")

                    if(remove_outlier == True):
                       
                        Q1 = data[column].quantile(0.25)
                        Q3 = data[column].quantile(0.75)

                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        data[column] = np.where(data[column] < lower_bound, np.nan, data[column])
                        data[column] = np.where(data[column] > upper_bound, np.nan, data[column])
                        
                        st.info('Outliers are handled of selected column')

                        st.subheader("Box plot after Removeing Outliers")
                        fig = px.box(data, y = column)
                        st.plotly_chart(fig)



    st.subheader(':rainbow[Outliers Handling]',divider='gray')
    remove_all_outlier = st.button("Remove Outliers of all Numerical Columns")         
    numerical_columns = list(data.select_dtypes(include=['int64', 'float64']).columns)
    categorial_columns = list(data.select_dtypes(include=['object']).columns)
    if(remove_all_outlier == True):
        data.drop_duplicates()
        def handle_outlier(data, column):          
                        Q1 = data[column].quantile(0.25)
                        Q3 = data[column].quantile(0.75)

                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        data[column] = np.where(data[column] < lower_bound, np.nan, data[column])
                        data[column] = np.where(data[column] > upper_bound, np.nan, data[column])

                        st.subheader(f"Boxplot for column '{column}' after handling outliers")
                        fig = px.box(data, y = column)
                        st.plotly_chart(fig)

        for col in numerical_columns:
            handle_outlier(data, col)

    
    # Handle Missing values 
    st.subheader(':rainbow[Handle Missing Values]',divider='rainbow')
    st.write("Handle Missing Values of Numerical Columns")
    numerical = st.button("Handle", key="Numerical Columns")
    if(numerical == True):
        for col in numerical_columns:
              data[col] = data[col].fillna(data[col].mean())
        st.info("Missing Values Handles Successfully for Numerical Columns")

    st.write("Handle Missing Values of Categorical Columns")
    categorical = st.button("Handle", key="Categorical Columns")
    if(categorical == True):
        for col in categorial_columns:
              data[col] = data[col].fillna(data[col].mode())
        st.info("Missing Values Handles Successfully for Categorical Columns")


     # Convert categorial values into numerical values
    st.subheader(':rainbow[Convert Categorical data into numerical data]',divider='gray')
    convert = st.button("Convert")
    if(convert == True):
        label_encoder = LabelEncoder()
        categorial_columns = list(data.select_dtypes(include=['object']).columns)
        for col in categorial_columns:
            data[col] = label_encoder.fit_transform(data[col])
        st.info("Successfully Converted")                    
    


    st.subheader(':rainbow[Standardized the attributes]', divider='gray')
    standrized = st.button("Standardized")
    if(standrized == True):
          scaler = StandardScaler()
          data[data.columns] = scaler.fit_transform(data[data.columns])
          st.info("Attributes are Standardized")

    # Apply train test aplit
    st.subheader(':rainbow[Apply Train Test Aplit]',divider='gray')
    target_column = st.selectbox("Select Target Column", options=data.columns)
    test_size = st.number_input('Test Data Size',key='slider', max_value=20, min_value=5)
    random_state = st.number_input('Random State', min_value=1)
    st.write(f"Target Column is '{target_column}'")
    st.write(f"Test Data Size is '{test_size}'")
    st.write(f"Random State value is '{random_state}'")
    apply = st.button("Apply Train Test Split")

    if(apply == True):
        for col in numerical_columns:
            data[col] = data[col].fillna(data[col].mean())
        for col in categorial_columns:
            data[col] = data[col].fillna(data[col].mode())  

        x = data.drop(target_column, axis=1)
        y = data[target_column]
        test_size = float(test_size / 100)
        st.write(test_size)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_state)
        st.info("Train Test Split is successfully applies.")


        st.subheader("View Data")
        view1,view2,view3, view4 = st.tabs(['X train','Y train', 'X test', 'Y Test'])

        with view1:
            st.dataframe(x_train)
        with view2:
            st.dataframe(y_train)
        with view3:
            st.dataframe(x_test)
        with view4:
            st.write(y_test)

        # lg = LinearRegression()
        # lg = lg.fit(x_train, y_train)
        # st.write("Linear Model applied")
        # predicted = lg.predict(x_test)
        # st.write("Predicted Y test is :")
        # st.dataframe(pd.DataFrame(predicted, columns=['Predicted Y']))

        st.subheader(':rainbow[Select model to apply on data]',divider='gray')
        # model = st.selectbox('Select', options=['Select', 'Linear Regression', 'Logistic Regression'])
        # st.write(f"Selected Model '{model}'")

        # if model == 'Linear Regression':
        st.subheader("Apply Linear Rgression")
        linear = st.button("Apply", key = "Apply Linear Regression")
        # if(linear == True):
        lg = LinearRegression()
        lg = lg.fit(x_train, y_train)
        st.info("Linear Regression applied")
        predicted = lg.predict(x_test)
        st.write("Predicted Y test is :")
        st.dataframe(pd.DataFrame(predicted, columns=['Predicted Y']))
        r2 = r2_score(y_test, predicted)
        mse = mean_squared_error(y_test, predicted)
        mae = mean_absolute_error(y_test, predicted)

        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        pickle_linear = st.button("Pickle File")

        # if(pickle_linear == True):
        linear_model_pkl_file = "linear_model.pkl"
        with open(linear_model_pkl_file, 'wb') as file:
                pickle.dump(lg, file)
                st.info("Model Successfully Pickled and saved in your System")
        with open(linear_model_pkl_file, "rb") as file:
            st.download_button(
            label="Download Pickle File",
            data=file,
            file_name="linear_model.pkl",
            mime="application/octet-stream"
        )

        st.write("Apply Logistic Regression")
        logistic = st.button("Apply", key="Apply Logistic Regression")
        # if(logistic == True):
        logerg = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
        logerg = logerg.fit(x_train, y_train)
        st.info("Logistic Regression applied")
        predicted = logerg.predict(x_test)
        st.write("Predicted Y test is :")
        st.dataframe(pd.DataFrame(predicted, columns=['Predicted Y']))
        cnf_matrix = metrics.confusion_matrix(y_test, predicted)
        st.write("Accuracy of the model is : ", metrics.accuracy_score(y_test, predicted) * 100)
            
        pickle_logistic = st.button("Pickle File")
        # if(pickle_logistic == True):
        logistic_model_pkl_file = "model.pkl"
        with open(logistic_model_pkl_file, 'wb') as file1:
                pickle.dump(logerg, file1)
                st.info("Model Successfully Pickled and saved in your System")
        with open(logistic_model_pkl_file, "rb") as file1:
            st.download_button(
            label="Download Pickle File",
            data=file1,
            file_name="logistic_model.pkl",
            mime="application/octet-stream"
        )          
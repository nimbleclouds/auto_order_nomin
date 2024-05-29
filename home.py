import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import hmac
import plotly.express as px
import seaborn as sns
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pyecharts.options as opts
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Line, Bar, Scatter, Boxplot

# # def check_password():
# #     def password_entered():
# #         if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
# #             st.session_state["password_correct"] = True
# #             del st.session_state["password"]
# #         else:
# #             st.session_state["password_correct"] = False

# #     # Return True if the password is validated.
# #     if st.session_state.get("password_correct", False):
# #         return True

# #     # Show input for password.
# #     st.text_input(
# #         "Password", type="password", on_change=password_entered, key="password"
# #     )
# #     if "password_correct" in st.session_state:
# #         st.error("Нууц үг буруу байна")
# #     return False


# if not check_password():
#     st.stop()
df = pd.read_csv('result_df.csv')
item_info = pd.read_csv('df1.csv')
df = df.drop(columns=df.columns[0])
df = df[['date','item_name','base_price','real_qty','ml_preds','S0','Q_auto','ML_auto','abs_err_Q','abs_err_ML']]
df.columns = ['ds','name','bprice','qty','ml_preds','auto_preds','order_auto','order_ml','mae_auto','mae_ml']

st.set_page_config(layout="wide")
st.title("Номин Юнайтед Хайпермаркетын барааны борлуулалт тооцоолох загварыг автомат захиалгатай харьцуулах практикал тест")
st.write("Бүх хувьсагч бүрэн, ширхэгийн бараанууд дээр суурилсан")



item = df.name.unique()
item_choices = st.selectbox('Бараа сонгох:',item)

result_df = df.copy()
result_df = result_df.merge(item_info, left_on='name',right_on='item_name',how='left').isna().sum()



df_sum = item_info.groupby(['item_name']).sum(numeric_only=True).reset_index()
df_avg = item_info.groupby(['item_name']).mean(numeric_only=True).reset_index()

item_info["weekday"] = item_info['date'].dt.dayofweek
item_info["weekend"] = item_info['date'].dt.dayofweek > 4
weekday = item_info.groupby(['item_name','weekday']).median().reset_index().set_index('weekday')[['item_name','qty']]


def generate_bar_chart(data, title):
    bar = (
        Bar()
        .add_xaxis(data.index.tolist())
        .add_yaxis("", data.tolist(), itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
            legend_opts=opts.LegendOpts(is_show=False)
        )
    )
    return bar


bt = st.button('Процесс хийх')


if bt:
    st.header(f"{item_choices}")
    st.divider()
    con = st.container()
    temp_1_1 = result_df[result_df['name']==item_choices]
    st.subheader('Барааны мэдээлэл')
    
    tab1_col1, tab1_col2 = st.columns(2)
    with tab1_col1:  
        st.write(f":red[Груп:] {item_info[item_info['item_name']==item_choices]['group'].values[0]}")
        st.write(f":red[Ангилал:] {item_info[item_info['item_name']==item_choices]['category'].values[0]}")
        st.write(f":red[Бренд:] {item_info[item_info['item_name']==item_choices]['brand'].values[0]}")
        st.write(f":red[Вендор:] {item_info[item_info['item_name']==item_choices]['vendor'].values[0]}")
    with tab1_col2:
        t1c2_total_amt = '₮'+format(df_sum[df_sum['item_name']==item_choices]['amt'].values[0],',.2f')
        st.write(f":red[Нийт борлуулалтын дүн:] {t1c2_total_amt}")
        t1c2_total_sale = format(df_sum[df_sum['item_name']==item_choices]['qty'].values[0],',.2f')
        st.write(f":red[Нийт борлуулсан:] {t1c2_total_sale}")
        t1c2_avg_price = '₮'+format(df_avg[df_avg['item_name']==item_choices]['base_price'].values[0],',.2f')
        st.write(f":red[Дундаж үнэ:] {t1c2_avg_price}")
        t1c2_avg_sale = format(df_avg[df_avg['item_name']==item_choices]['qty'].values[0],',.2f')
        st.write(f":red[Өдрийн дундаж борлуулалтын тоо:] {t1c2_avg_sale}")
        
        
        
    st.divider()
    
    
    
    tab1_col3, tab1_col4 = st.columns(2)
    with tab1_col3:
        st.subheader(f":white[Борлуулалтын тоо:]")
        temp_1_1 = item_info[item_info['item_name']==item_choices]
        temp_1_1 = temp_1_1.set_index('date')
        sales_monthly_1 = temp_1_1.resample('M').sum()
        sales_monthly_1.index = sales_monthly_1.index.strftime('%Y-%m')
        line = (
            Line()
            .add_xaxis(sales_monthly_1.index.tolist())  # Convert the index to a list
            .add_yaxis(series_name='Борлуулалтын тоо', y_axis=sales_monthly_1['qty'].tolist(), is_smooth=True,
                       linestyle_opts=opts.LineStyleOpts(color="red"),
                       label_opts=opts.LabelOpts(is_show=False),
                       symbol="none",  # Remove data point symbols
                       itemstyle_opts=opts.ItemStyleOpts(color="red"))
            .set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                      axisline_opts=opts.AxisLineOpts(is_show=False),
                                                      axistick_opts=opts.AxisTickOpts(is_show=False),
                                                      splitline_opts=opts.SplitLineOpts(is_show=False)),
                             yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                      axisline_opts=opts.AxisLineOpts(is_show=False),
                                                      axistick_opts=opts.AxisTickOpts(is_show=False),
                                                      splitline_opts=opts.SplitLineOpts(is_show=False)),
                             tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                           formatter=opts.TooltipOpts(formatter='{b}: {c}')))
        )
        st_pyecharts(line)
        
        
        #fig = px.line(sales_monthly_1['qty'],line_shape="spline")
        #fig.update_traces(line_color='#FF0000',line_width=2.4,showlegend=False)
        #fig.update_layout(xaxis_title='Огноо', yaxis_title='Тоо')
        #st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f":white[7 хоногийн өдөр тус бүрийн дундаж борлуулалт:]")
        temp_1_2 = weekday[weekday['item_name']==item_choices].copy()
        
        bar = (
        Bar()
        .add_xaxis(temp_1_2.index.tolist())  # Convert the index to a list
        .add_yaxis(series_name='Борлуулалтын тоо', y_axis=temp_1_2['qty'].tolist(),
               label_opts=opts.LabelOpts(is_show=False),  # Remove data point symbols
               itemstyle_opts=opts.ItemStyleOpts(color="red"))
        .set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                              axisline_opts=opts.AxisLineOpts(is_show=False),
                                              axistick_opts=opts.AxisTickOpts(is_show=False),
                                              splitline_opts=opts.SplitLineOpts(is_show=False)),
                     yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                              axisline_opts=opts.AxisLineOpts(is_show=False),
                                              axistick_opts=opts.AxisTickOpts(is_show=False),
                                              splitline_opts=opts.SplitLineOpts(is_show=False)),
                     tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                   formatter=opts.TooltipOpts(formatter='{b}: {c}')),
                        legend_opts=opts.LegendOpts(is_show=False))
        )
        st_pyecharts(bar)
        
        #fig = px.bar(temp_1_2['qty'],text = temp_1_2["qty"])
        #fig.update_traces(marker_color='#FF0000',showlegend=False)
        #fig.update_layout(xaxis_title='Гараг', yaxis_title='Тоо')
        #st.plotly_chart(fig,use_container_width=True)
        
    with tab1_col4:
        st.subheader(f":white[Борлуулалтын тренд:]")
        # Assuming eda is your DataFrame and item_choices is a list of item names
        k_1 = item_info[item_info['item_name'] == item_choices]
        k_1 = k_1.set_index('date')
        k_1 = k_1.resample('M').mean(numeric_only=True)
        numeric_dates_1 = (k_1.index - k_1.index[0]).days
        coefficients = np.polyfit(numeric_dates_1, k_1['qty'], 1)
        trendline_x = numeric_dates_1  # Use original x-axis data for the trendline
        trendline_y = np.polyval(coefficients, numeric_dates_1)  # Calculate predicted y-values for the trendline

        # Convert datetime index to strings with format 'YYYY-MM'
        x_labels = k_1.index.strftime('%Y-%m').tolist()

        # Create the Scatter chart
        scatter = (
            Scatter()
            .add_xaxis(x_labels)  # Set x-axis data with formatted dates
            .add_yaxis("Борлуулалтын тоо", k_1['qty'].tolist(), label_opts=opts.LabelOpts(is_show=False), symbol="circle", 
                       symbol_size=8, itemstyle_opts=opts.ItemStyleOpts(color="red"))  # Set y-axis data with a series name and red color
        )

        # Create the Line chart for the trendline
        line = (
            Line()
            .add_xaxis(x_labels)  # Set x-axis data for the trendline with formatted dates
            .add_yaxis("Хандлагын шугам", trendline_y.tolist(), 
                       linestyle_opts=opts.LineStyleOpts(color="white", type_='dotted',width=0.5),  # Turn the line red
                       symbol="none",itemstyle_opts=opts.ItemStyleOpts(color="white"))  # Remove markers
        )

        # Combine the Scatter and Line charts
        scatter.overlap(line)
        #a
        # Set global options for the chart
        scatter.set_global_opts(
                                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                                         axistick_opts=opts.AxisTickOpts(is_show=False),
                                                         splitline_opts=opts.SplitLineOpts(is_show=False)),
                                yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"),
                                                         axisline_opts=opts.AxisLineOpts(is_show=False),
                                                         axistick_opts=opts.AxisTickOpts(is_show=False),
                                                         splitline_opts=opts.SplitLineOpts(is_show=False)),
                                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                              formatter=opts.TooltipOpts(formatter='{b}: {c}'))
                               )

        st_pyecharts(scatter)
        
        
        
        st.subheader(f":white[Борлуулагдах тоо хэмжээний дистрибюшн:]")
        f_1 = item_info[item_info['item_name']==item_choices]
        f_1 = f_1.groupby(['date','item_name']).sum('qty').reset_index()
        fig = px.violin(f_1['qty'])
        fig.update_traces(marker_color='#FF0000',showlegend=False)
        fig.update_layout(xaxis_title='',yaxis_title='',height=350)
        st.plotly_chart(fig, use_container_width=True)
        

    st.divider()
    
    
    
    #'qty','ml_preds','auto_preds','order_auto','order_ml','mae_auto','mae_ml'
    st.write('Борлуулалтын таамаглал буюу S0 хувьсагчийн харьцуулалт')
    fcst_1 = result_df[result_df.item_name==item_choices].set_index('ds')[['qty','ml_preds','auto_preds']]
    fcst_1 = fcst_1.rename(columns={'auto_preds':'Автомат',
                    'qty':'Бодит',
                    'ml_preds':'МЛ'})
    forecast1 = (
    Line()
    .add_xaxis(fcst_1.index.tolist())
    .add_yaxis("Автомат захиалга", fcst_1['Автомат'].tolist(), linestyle_opts=opts.LineStyleOpts(color="orange", type_="dashed"),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="orange"))
    .add_yaxis("Бодит", fcst_1['Бодит'].tolist(), linestyle_opts=opts.LineStyleOpts(color="red", type_="solid", width=2),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="red"))
    .add_yaxis("Загвар", fcst_1['МЛ'].tolist(), linestyle_opts=opts.LineStyleOpts(color="white", type_="dashed"),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="white"))
    .set_global_opts(
        title_opts=opts.TitleOpts(title=""),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
        legend_opts=opts.LegendOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                           formatter=opts.TooltipOpts(formatter='{b}: {c}'))))

    
    st_pyecharts(forecast1)
    
    st.write('Захиалах тооны тооцоолол харьцуулалт')
        
    fcst_2 = result_df[result_df.item_name==item_choices].set_index('ds')[['order_auto','order_ml']]
    fcst_2 = fcst_2.rename(columns={'order_auto':'Автомат',
                    'order_ml':'МЛ'})
    forecast2 = (
    Line()
    .add_xaxis(fcst_2.index.tolist())
    .add_yaxis("Автомат захиалга", fcst_2['Автомат'].tolist(), linestyle_opts=opts.LineStyleOpts(color="orange", type_="dashed"),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="orange"))
    .add_yaxis("Загвар", fcst_2['МЛ'].tolist(), linestyle_opts=opts.LineStyleOpts(color="white", type_="dashed"),symbol="none", label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="white"))
    .set_global_opts(
        title_opts=opts.TitleOpts(title=""),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(color="white"), splitline_opts=opts.SplitLineOpts(is_show=False)),
        legend_opts=opts.LegendOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross",
                                                           formatter=opts.TooltipOpts(formatter='{b}: {c}'))))

    
    st_pyecharts(forecast2)
    
    st.write('S0 хувьсагчийн бодит дүнтэй харьцуулсан зөрүү')
    #'mae_auto','mae_ml'
    lossmetrics = result_df[result_df.item_name==item_choices].rename(columns={'mae_auto':'Автомат захиалгын алдагдлын дүн',
                                                                  'mae_ml':'МЛ алдагдлын дүн'})
    lossamt = lossmetrics[['Автомат захиалгын алдагдлын дүн', 'Загварын алдагдлын дүн']]

    st.write("Зөрүү")
    bar_loss_err = generate_bar_chart(lossamt.squeeze(), "")
    st_pyecharts(bar_loss_err)
    
                                                                               
    filtered_df = result_df[result_df.item_name==item_choices]                                                   
    filtered_df = result_df[result_df['qty'] != 0].copy()
    filtered_df['abs_perc_err_Q'] = abs((filtered_df['qty'] - filtered_df['auto_preds']) / (filtered_df['qty']))
    filtered_df['abs_perc_err_ML'] = abs((filtered_df['qty'] - filtered_df['ml_preds']) / (filtered_df['qty']))

    # Drop intermediate columns if needed
    filtered_df.drop(columns=['abs_perc_err_Q', 'abs_perc_err_ML'], inplace=True)
    st.write(f"MAPE (Автомат): {mape_Q*100}%")
    st.write(f"MAPE (МЛ): {mape_ML*100}%")

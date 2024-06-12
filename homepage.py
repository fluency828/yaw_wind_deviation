import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import matplotlib.dates as mdate
from matplotlib import rcParams
import matplotlib as mpl
from utils import save_data,save_figures
import matplotlib as mpl
import io
import openpyxl
from site_function import wind_power_plant
# import missingno as msno

mpl.font_manager.fontManager.addfont('字体/SIMSUN.ttf')
config = {
    "font.family":'serif',
    # "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SIMSUN'],
}
rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False


########################## 正式开始网页！###################

st.title('偏航对风偏差预警 通用调参工具')
wtg_pn = st.sidebar.text_input("请输入风机号测点名称",'device_name')
time_pn = st.sidebar.text_input("请输入时间测点名称",'timestamp_utc')
w_pn = st.sidebar.text_input("请输入风速测点名称",'windspeed')
yaw_angle_pn = st.sidebar.text_input("请输入机舱与风向夹角测点名称","ai00048")
P_pn = st.sidebar.text_input("请输入有功功率测点名称",'genactivepw')
temp_pn = st.sidebar.text_input("请输入机舱温度测点名称",'temout')
wd_pn = st.sidebar.text_input("请输入风向测点名称",'winddirection')
genspd_pn = st.sidebar.text_input("请输入发电机转速测点名称",'genspd')
blade1_pn = st.sidebar.text_input("请输入桨叶1测点名称",'blade1position')
blade2_pn = st.sidebar.text_input("请输入桨叶2测点名称",'blade2position')
blade3_pn = st.sidebar.text_input("请输入桨叶3测点名称",'blade3position')
blade_pn = st.sidebar.text_input("请输入桨叶平均测点名称",'blade_average')
Cp_pn = st.sidebar.text_input("请输入功率系数测点名称",'CP')
group_by = st.sidebar.selectbox('选择是否按分钟聚合数据',[True,False])
rated_power =  float(st.sidebar.text_input("请输入额定功率",2500))

raw_data_path = st.file_uploader('上传原始数据(单个风机的秒级数据)')
# pw_cur_path = st.file_uploader('上传理论功率数据（包含机型）')


def load_data(url):
    df = pd.read_csv(url)
    return df


raw_data = load_data(raw_data_path)

wtg_id = np.unique(raw_data[wtg_pn])
assert len(wtg_id) == 1
st.write(raw_data)

site_instance = wind_power_plant(wtg_data=raw_data,
                                 wtg_id=wtg_id[0],
                                 theory_data=None,
                                 wtg_pn=wtg_pn,
                                 time_pn=time_pn,
                                 w_pn=w_pn,
                                 yaw_angle_pn=yaw_angle_pn,
                                 P_pn=P_pn,
                                 temp_pn=temp_pn,
                                 wd_pn=wd_pn,
                                 genspd_pn=genspd_pn,
                                 blade1_pn=blade1_pn,
                                 blade2_pn=blade2_pn,
                                 blade3_pn=blade3_pn,
                                 rated_power=rated_power,
                                 groupby=group_by,
                                 three_blade_pn = True if blade1_pn!=blade2_pn else False
                                 )

# theory_pw_cur = pd.read_excel(pw_cur_path)
st.markdown('# 查看原始数据')
st.write('wtg_data')
st.write(site_instance.wtg_data)

st.write('wtg_data Describe')
st.write(site_instance.wtg_data.describe())

st.write('month list')
least_amount = st.slider('选择每个月最少的数据量',0,2000,100,1600)
site_instance._get_month_list(least_amount)
if group_by:
    site_instance._groupby_minutes()
else:
    site_instance.wtg_data_1min = site_instance.wtg_data
st.write(site_instance.month_list)


st.write('# 查看不同月份的原始数据情况')
st.pyplot(site_instance._check_month_plot(site_instance.wtg_data))
if group_by:
    st.markdown('# 查看聚合后数据')
    st.write('wtg_1min_data')
    st.write(site_instance.wtg_data_1min)
    st.write('查看不同月份的聚合数据情况')
    st.pyplot(site_instance._check_month_plot(site_instance.wtg_data_1min))
st.write('缺失值')
st.pyplot(site_instance.show_na(site_instance.wtg_data_1min))
st.markdown('# 缺失值处理')
if_backfill = st.selectbox('是否向前填充桨叶角度缺失值',[True,False])
site_instance._fill_na(backfill=if_backfill)
st.write('缺失值填充后')
st.pyplot(site_instance.show_na(site_instance.wtg_data_1min))
st.markdown('# 工况处理')

columns = st.columns(3)

with columns[0]:
    blade_up = st.slider('图2的x轴右侧最大值',0,5,1,1)
with columns[1]:
    bin_size = st.slider('图2分bin数量',0,100,100,10)
with columns[2]:
    blade_scatter_up = st.slider('图3x轴右侧最大值',5,20,10,5)

st.pyplot(site_instance.draw_blade_wind(site_instance.wtg_data_1min,
                                        blade_uperbond=blade_up,
                                        bs=bin_size,
                                        blade_scatter_uperbond=blade_scatter_up))

columns = st.columns(4)

with columns[0]:
    lo = st.slider('爬坡风速最小值',0,20,4,1)
with columns[1]:
    up = st.slider('爬坡风速最大值',0,20,8,1)
with columns[2]:
    blade_th = st.slider('桨叶片角度限电阈值',-1,5,1,1)
with columns[3]:
    power_blade_change = st.slider('满发前变桨功率阈值',100,10000,10000,100)
site_instance.working_condition(power_blade_cha=power_blade_change,
                                blade_thre=blade_th,
                                wind_thre=[lo,up])
st.pyplot(site_instance._check_month_plot(site_instance.wtg_data_1min,'working_condition'))
st.write('工况处理数据结果')
st.write(site_instance.wtg_normal_power)

st.markdown('# 异常数据处理')
st.pyplot(site_instance.check_Cp())

columns = st.columns(2)

with columns[0]:
    if_diff = st.selectbox('是否进行防异常影响处理（差分）',[True,False])
with columns[1]:
    diff_thre = st.slider('处理阈值',min_value = 100,max_value = 2000,value = 1000,step = 100)
columns = st.columns(2)
with columns[0]:
    if_sigma = st.selectbox('是否进行3sigma分仓风速处理',[True,False])
with columns[1]:
    sigma_times = st.slider('3sigma异常值剔除次数',1,10,2,1)

columns = st.columns(3)
with columns[0]:
    if_Cp = st.selectbox('是否进行相对Cp异常值处理',[True,False])
with columns[1]:
    Cp_lo = st.text_input('Cp范围最小值',1.5,)
with columns[2]:
    Cp_up = st.text_input('Cp范围最大值',6,)


site_instance.clean_data(difference=if_diff,
                         difference_thre=diff_thre,
                         sigma_3=if_sigma,
                         sigma_times=sigma_times ,
                         Cp=if_Cp,
                         Cp_thre=[eval(Cp_lo),eval(Cp_up)])
st.pyplot(site_instance._check_month_plot(site_instance.wtg_normal_power,'nan_marker'))
st.markdown('# 对风偏差拟合')
columns = st.columns(4)
with columns[0]:
    least_samples =  st.slider('最小样本数量',1000,3000,1500,500)
with columns[1]:
    angle_lo = eval(st.text_input('最小角度',-30,))
with columns[2]:
    angle_up = eval(st.text_input('最大角度',30,))
with columns[3]:
    bin_length = eval(st.text_input('bin区间大小',0.5,))
site_instance.drop_yaw_outlier(yaw_angle_lo=angle_lo,yaw_angle_hi=angle_up)
site_instance._divide_float_to_bin(bin_len=bin_length)
columns = st.columns(2)
with columns[0]:
    q =  eval(st.text_input('分位数',0.5))
with columns[1]:
    covariant = st.selectbox('method used to calculate the variance-covariance matrix',['robust','iid'])
columns = st.columns(2)
with columns[0]:
    kernel = st.selectbox('kernel to use in the kernel density estimation for the asymptotic covariance matrix',['epa','cos','gau','par'])
with columns[1]:
    bwidth = st.selectbox('Bandwidth selection method in kernel density',['hsheather','bofinger','chamberlain'])

columns = st.columns(2)
with columns[0]:
    max_iter = eval(st.text_input('最大迭代次数',5000,))
with columns[1]:
    tolerance = eval(st.text_input('tolerance',1e-6,))

figure_list,table = site_instance.quantreg(least_samples,
                                           q,
                                           covariant,
                                           kernel,
                                           bwidth,
                                           max_iter,
                                           tolerance)
# col_ls = st.columns(4)
for i,figs in enumerate(figure_list):
    # with col_ls[i%4]:
    st.pyplot(figs)
st.write(table)

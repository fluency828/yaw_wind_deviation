import pandas as pd
import numpy as np
from scipy.stats import kstest
import missingno as msno
import math

import matplotlib.pyplot as plt
# from sklearn.linear_model import QuantileRegressor
import statsmodels.formula.api as smf

import matplotlib as mpl
from matplotlib import rcParams
import sys
sys.path.append('D:/OneDrive - CUHK-Shenzhen/utils/')
from xintian.power_limited import limit_power_detect_loc


mpl.font_manager.fontManager.addfont('D:/1 新天\数字运营部 任务\字体/SIMSUN.ttf')
config = {
    "font.family":'serif',
    # "font.size": 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SIMSUN'],
}
rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False

########################################################################
# 设置基本参数
########################################################################


ROOT_PATH = 'D:/1 新天\数字运营部 任务\逻辑文档\昆仑偏航对风逻辑文档/'
wtg_data = pd.read_csv(ROOT_PATH + '导出数据/a_ys_1q_09.csv');wtg_data

theory_pw_cur = pd.read_excel('D:/1 新天\数字运营部 任务\逻辑文档\昆仑偏航对风逻辑文档\SZ164-月度分析报告-偏航对风\元山试跑\理论功率曲线.xlsx')
theory_pw_cur

wtg_pn = 'device_name'
time_pn = 'timestamp_utc'
w_pn = 'windspeed'
yaw_angle_pn = 'ai00048'
P_pn = 'genactivepw'
temp_pn = 'temout'
wd_pn = 'winddirection'
genspd_pn = 'genspd'
blade1_pn = 'blade1position'
blade2_pn = 'blade2position'
blade3_pn = 'blade3position'
blade_pn = 'blade_average'
Cp_pn = 'CP'
########################################################################
# 读取数据并处理数据格式
########################################################################


wtg_data = wtg_data[[wtg_pn,time_pn,w_pn,yaw_angle_pn,\
                    P_pn,temp_pn,wd_pn,genspd_pn,\
                    blade1_pn,blade2_pn,blade3_pn]]


wtg_data[time_pn] = pd.to_datetime(wtg_data[time_pn])
wtg_data['month'] = wtg_data[time_pn].dt.month + wtg_data[time_pn].dt.year*100


### 更改数据类型
wtg_data = wtg_data.replace({'\\N':np.NaN})
for column in [w_pn,yaw_angle_pn,\
                P_pn,temp_pn,wd_pn,genspd_pn,\
                blade1_pn,blade2_pn,blade3_pn]:
    wtg_data[column] = wtg_data[column].astype(float)

wtg_data.describe()

########################################################################
# 获取 month_list
########################################################################

wtg_data = wtg_data[wtg_data['month']!=202401].reset_index(drop=True)
month_list = np.unique(wtg_data['month'])


# 查看各个month的数据情况
import math
nrows = math.floor(len(month_list)/3)
fig,axes = plt.subplots(nrows,3,figsize = (30,20))
for i,m in enumerate(month_list):
    m_data = wtg_data[wtg_data['month']==m].reset_index(drop=True)
    # print(f'{m}月数据大小为{m_data.shape}')
    axes[i//3][i%3].scatter(x=m_data[w_pn],y=m_data[P_pn],s=5)
    axes[i//3][i%3].set_xlim(0,30)
    axes[i//3][i%3].set_ylim(0,2200)
    axes[i//3][i%3].set_title(f'{m}月数据{m_data.shape}',fontsize=20)

########################################################################
# 数据按分钟聚合 
########################################################################

#（选择在处理缺失值之前先聚合，由于不是每一分钟都有数据，会产生大量缺失值）
wtg_data_1min = wtg_data.set_index(time_pn).groupby(wtg_pn).resample(rule='1T').mean().reset_index()

wtg_data_1min = wtg_data_1min.loc[~wtg_data_1min['month'].isnull(),:].reset_index(drop=True)
print('聚合后数据',wtg_data_1min.shape)

# wtg_data.loc[(wtg_data[time_pn]>='2023-07-01')&(wtg_data[time_pn]<='2023-07-02'),:].\
#     sort_values(by=time_pn).to_excel(ROOT_PATH+'查看分钟聚合后的nan从哪来.xlsx',index=False)
# 结果发现该分钟的数据缺失是由于 没有该分钟的秒级数据
# 这部分数据无法填充

month2_dict = {}
for i in range(len(month_list)):
    # print(i,month_list[i])
    if i % 2 == 0:
        month2_dict[month_list[i]] = f'{month_list[i]},{month_list[i+1]}'
    if i % 2 == 1:
        month2_dict[month_list[i]] = f'{month_list[i-1]},{month_list[i]}'
wtg_data_1min['2month'] = wtg_data_1min['month'].replace(month2_dict)
month2_list = np.unique(wtg_data_1min['2month'])

########################################################################
# 缺失值处理
########################################################################

## 查看缺失值
msno.matrix(wtg_data_1min)
# print(wtg_data_1min[[time_pn,blade1_pn,blade2_pn,blade3_pn]].describe())

## 填充桨叶角度缺失值，功率缺失直接删除
wtg_data_1min = wtg_data_1min.loc[~wtg_data_1min[P_pn].isnull(),:].reset_index(drop=True)
wtg_data_1min = wtg_data_1min.sort_values(by=[wtg_pn,time_pn]).reset_index(drop=True)
wtg_data_1min[blade_pn] = wtg_data_1min[[blade1_pn,blade2_pn,blade3_pn]].min(axis=1)

wtg_data_1min[blade_pn] = wtg_data_1min[[wtg_pn,blade_pn]].groupby([wtg_pn]).fillna(method = 'backfill')
# print(wtg_data_1min[[time_pn,blade1_pn,blade_pn]].describe())
print('缺失值处理后数据',wtg_data_1min.shape)


########################################################################
# 去除功率曲线周围的限电、异常数据等
########################################################################

def KsNormDetect(df,pn,if_plot = False,title=None):
    # 计算均值
    u = df[pn].mean()
    # 计算标准差
    std = df[pn].std()
    # 计算P值
    print(kstest(df[pn], 'norm', (u, std)))
    res = kstest(df[pn], 'norm', (u, std))[1]
    print('均值为：%.2f，标准差为：%.2f' % (u, std))
    # 判断p值是否服从正态分布，p<=0.05 拒绝原假设 不服从正态分布
    if res <= 0.05:
        print('该列数据不服从正态分布')
        print("-" * 66)
        if if_plot:
            fig,ax = plt.subplots(figsize = (16,8))
            _=ax.hist(x=df[pn],bins=500)
            ax.set_title(f'{title}不符合正态分布')
        return True
    else:
        print('该列数据服从正态分布')
        if if_plot:
            fig,ax = plt.subplots(figsize = (16,8))
            _=ax.hist(x=df[pn],bins=500)
            ax.set_title(f'{title}符合正态分布')
        print("-" * 66)
        return False

def OutlierDetection(df, ks_res,pn):
    # 计算均值
    u = df[pn].mean()
    # 计算标准差
    std = df[pn].std()
    if ks_res is not None:
        # 定义3σ法则识别异常值
        outliers = df[np.abs(df[pn] - u) > 3* std].index
        # 剔除异常值，保留正常的数据
        outliers_portion = round(len(outliers)/df.shape[0],4)
        print(f'异常值有{len(outliers)},占比{outliers_portion*100}%')
        # df.loc[outliers,pn] = np.NaN
        # 返回异常值和剔除异常值后的数据
        return outliers

    else:
        print('请先检测数据是否服从正态分布')
        return None


####### 根据CP去除异常值
wtg_data_1min[Cp_pn] = wtg_data_1min[P_pn]/np.where(wtg_data_1min[w_pn]<1,1,wtg_data_1min[w_pn]**3)

# 画图查看Cp值分布
fig,ax = plt.subplots(figsize = (16,8))
# ax.scatter(x=wtg_data_1min[blade1_pn],y=wtg_data_1min[P_pn])
_ = ax.hist(x=wtg_data_1min[Cp_pn],bins=600)
ax.set_title('Cp的分布情况',size=20)

# ks = KsNormDetect(wtg_data_1min,Cp_pn)
# o_index = OutlierDetection(wtg_data_1min,ks,Cp_pn)
# Cp_out_data = wtg_data_1min.iloc[o_index,:].reset_index(drop=True)
# wtg_data_1min = wtg_data_1min.drop(o_index).reset_index(drop=True)
# print('Cp 3σ 筛选后数据',wtg_data_1min.shape)

Cp_out_data = wtg_data_1min[(wtg_data_1min[Cp_pn]<=0)|(wtg_data_1min[Cp_pn]>10)].reset_index(drop=True)

wtg_data_1min = wtg_data_1min[(wtg_data_1min[Cp_pn]>0)&(wtg_data_1min[Cp_pn]<=10)].reset_index(drop=True)
print('Cp 3σ 筛选后数据',wtg_data_1min.shape)

####### 根据桨叶角度去除异常值

fig,ax = plt.subplots(figsize = (16,8))
# ax.scatter(x=wtg_data_1min[blade1_pn],y=wtg_data_1min[P_pn])
_=ax.hist(x=wtg_data_1min[wtg_data_1min[blade_pn]<0][blade_pn],bins=500)
ax.set_title('桨叶角度分布',size=20)


blade_out_data = wtg_data_1min[wtg_data_1min[blade_pn]>-0.4].reset_index(drop=True)

wtg_data_1min = wtg_data_1min[wtg_data_1min[blade_pn]<=-0.4].reset_index(drop=True)
wtg_data_1min = wtg_data_1min[wtg_data_1min[blade_pn]>=-0.52].reset_index(drop=True)

print('桨叶角度筛选后数据',wtg_data_1min.shape)


########################################################################
# 风速筛选爬坡阶段，并分仓
########################################################################
# 查看有功功率-风速散点图，以选择合适的风速范围（爬坡阶段）

fig,axes = plt.subplots(figsize = (20,8))
axes.scatter(wtg_data[w_pn],wtg_data[P_pn])
axes.set_xlim(0,30)
axes.set_ylim(0,2200)

wtg_data_wind= wtg_data_1min[wtg_data_1min[w_pn]>=3.75].reset_index(drop=True)
wtg_data_wind = wtg_data_wind[wtg_data_wind[w_pn]<8.25].reset_index(drop=True)

fig,axes = plt.subplots(figsize = (20,8))
axes.scatter(wtg_data_wind[w_pn],wtg_data_wind[P_pn])
axes.set_xlim(0,30)
axes.set_ylim(0,2200)

# bs = np.arange(3.75,8.5,0.5);bs
# len(bs)
# ls = np.arange(4,8.5,0.5);ls
# len(ls)

bs = np.arange(3.75,8.35,0.1);bs
len(bs)
ls = np.linspace(3.8,8.2,45);ls
len(ls)


wtg_data_wind['wind_bin'] = pd.cut(wtg_data_wind[w_pn],bins=bs,right=False,labels=ls).astype(float)
# wtg_data_1min = wtg_data_1min[wtg_data_1min['wind_bin'].notnull()].reset_index(drop=True)
print('风速筛选后数据',wtg_data_wind.shape)



####### 固定风速仓对功率进行3σ准则缩尾

nrows = math.floor(len(month_list)/3)
fig,axes = plt.subplots(nrows,3,figsize = (30,20))
index_list = []
for i,m in enumerate(month_list):
    month_data = wtg_data_wind[wtg_data_wind['month']==m]
    month_index_list = []
    for w in ls:
        print(f'风速仓{w}m/s')
        w_data = month_data[month_data['wind_bin'] == w]
        ks = KsNormDetect(w_data,P_pn,False)
        o_index = OutlierDetection(w_data,ks,P_pn)
        index_list += list(o_index)
        month_index_list += list(o_index)
    sigma3_out_month_data = wtg_data_wind.iloc[month_index_list,:]
    axes[i//3][i%3].scatter(x=month_data[w_pn],y=month_data[P_pn],s=5)
    axes[i//3][i%3].scatter(x=sigma3_out_month_data[w_pn],y=sigma3_out_month_data[P_pn],s=10,label='功率sigma3')
    axes[i//3][i%3].set_xlim(0,30)
    axes[i//3][i%3].set_ylim(0,2200)
    axes[i//3][i%3].set_title(f'{m}月数据{month_data.shape}',fontsize=20)

sigma3_out_data = wtg_data_wind.iloc[index_list,:]
wtg_data_wind = wtg_data_wind.drop(index_list).reset_index(drop=True)
print('功率3σ（固定风速）筛选后数据',wtg_data_wind.shape)




####### 画图
fig,axes = plt.subplots(1,2,figsize = (24,8))
axes[0].scatter(wtg_data_1min[w_pn],wtg_data_1min[P_pn],label='原始数据',s=3)
axes[0].scatter(blade_out_data[w_pn],blade_out_data[P_pn],label='桨叶角度剔除数据',s=3)
axes[0].scatter(sigma3_out_data[w_pn],sigma3_out_data[P_pn],label='功率3σ剔除数据',s=3)
axes[0].scatter(Cp_out_data[w_pn],Cp_out_data[P_pn],label='Cp 剔除数据',s=3)
axes[0].set_xlim(0,30)
axes[0].set_ylim(0,2200)
axes[0].legend()

axes[1].scatter(wtg_data_wind[w_pn],wtg_data_wind[P_pn],label='最终使用数据')
axes[1].set_xlim(0,30)
axes[1].set_ylim(0,2200)
axes[1].legend()




# 查看当前功率分布,决定是否需要对功率进行处理
fig,ax = plt.subplots(figsize = (16,8))
_=ax.hist(x=wtg_data_wind[P_pn],bins=500)
wtg_data_wind = wtg_data_wind[wtg_data_wind[P_pn]>0].reset_index(drop=True)

# 传统去除限电方法

# gen_data,raw_data_1 = limit_power_detect_loc(wtg_data_1min,theory_pw_cur,\
#                                             wtg_pn=wtg_pn,time_pn=time_pn,\
#                                             wind_pn=w_pn,P_pn=P_pn,\
#                                             blade_angle_pn=blade1_pn,angle_thr=0,\
#                                             gap_thr=0.01,pw_thr=0,multiple_type=False)


########################################################################
# 偏航角度筛选[-30,30]
########################################################################

wtg_data_wind = wtg_data_wind[wtg_data_wind[yaw_angle_pn] <= 30].reset_index(drop=True)
wtg_data_wind = wtg_data_wind[wtg_data_wind[yaw_angle_pn] >= -30].reset_index(drop=True)

########################################################################
# 查看每两个月的功率-风速散点图
########################################################################
# reg.coef_
import math
nrows = math.floor(len(month2_list)/3)
fig,axes = plt.subplots(nrows,3,figsize = (30,10))
for i,m in enumerate(month2_list):
    m_data = wtg_data_wind[wtg_data_wind['2month']==m].reset_index(drop=True)
    # print(f'{m}月数据大小为{m_data.shape}')
    axes[i%3].scatter(x=m_data[w_pn],y=m_data[P_pn],s=5)
    axes[i%3].set_xlim(0,30)
    axes[i%3].set_ylim(0,2200)
    axes[i%3].set_title(f'{m}月数据{m_data.shape}',fontsize=20)
    
########################################################################
# 中位数拟合，并画图
########################################################################

bs = np.arange(3.75,8.5,0.5);bs
len(bs)
ls = np.arange(4,8.5,0.5);ls
len(ls)

wtg_data_wind['wind_bin'] = pd.cut(wtg_data_wind[w_pn],bins=bs,right=False,labels=ls).astype(float)

 



h_list = []
m_list = []
for m in month2_list:
    m_data = wtg_data_wind[wtg_data_wind['2month']==m].reset_index(drop=True)
    print(f'{m}月数据大小为{m_data.shape}')
    if m_data.shape[0]==0:
        continue
    fig,axes = plt.subplots(3,3,figsize = (20,20))
    mh_list = []
    m_list.append(m)
    for i,v in enumerate(ls):
        v_data = m_data[m_data['wind_bin']==v].reset_index(drop=True)
        # print(f'数据大小为{v_data.shape}')
        v_data['x1'] = v_data[yaw_angle_pn]
        v_data['x2'] = v_data[yaw_angle_pn]**2
        v_data['y'] = v_data[P_pn]
        X = v_data[['x1','x2']]
        y = v_data['y']
        reg = smf.quantreg("y ~ x1 + x2",v_data)
        res = reg.fit(q=0.5)
        # print(res.summary())
        preds = res.predict(X)
        p1 = res.params[1]
        p2 = res.params[2]
        h = -p1/(2*p2)
        axes[i//3][i%3].scatter(x=v_data[yaw_angle_pn],y=v_data[P_pn],s=5)
        axes[i//3][i%3].scatter(v_data[yaw_angle_pn],preds,color='red',s=0.5)
        axes[i//3][i%3].set_title(f'{m}月 {v}m/s 极值点{round(h,2)} \n数据量{v_data.shape[0]}',\
                     fontsize=20)
        if (abs(h)<30) and (v_data.shape[0]>2500):
            mh_list.append(h)
        else:
            # print(abs(h),v_data.shape[0])
            # print((abs(h)<30),v_data.shape[0]>2000)
            mh_list.append(np.NaN)
        # print(mh_list)
    h_list.append(mh_list)
    plt.savefig(ROOT_PATH + f'{m}月.jpg',bbox_inches='tight',facecolor='white',dpi=500)

result = pd.DataFrame(h_list).T
result.columns = m_list
result[w_pn] = ls

results_ls = []
for m in m_list:
    result_m = result[[m,w_pn]].dropna().reset_index(drop=True)
    angle = (result_m[m]*(result_m[w_pn]**3)).sum()/(result_m[w_pn]**3).sum()
    results_ls.append(angle)



print(results_ls)


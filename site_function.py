import numpy as np
import pandas as pd
import sys
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import missingno as msno
import sys
import os
import math
import seaborn as sns
import missingno
import statsmodels.formula.api as smf


class wind_power_plant():
    def __init__(self,
                wtg_data,
                wtg_id,
                theory_data,
                wtg_pn = 'device_name',
                time_pn = 'timestamp_utc',
                w_pn = 'windspeed',
                yaw_angle_pn = 'ai00048',
                P_pn = 'genactivepw',
                temp_pn = 'temout',
                wd_pn = 'winddirection',
                genspd_pn = 'genspd',
                blade1_pn = 'blade1position',
                blade2_pn = 'blade2position',
                blade3_pn = 'blade3position',
                # blade_pn = 'blade_average',
                # Cp_pn = 'CP'
                rated_power = 2500,
                groupby=True,
                three_blade_pn = True
                ) -> None:
        blade_pn_list = [blade1_pn,blade2_pn,blade3_pn] if three_blade_pn else [blade1_pn]
        wtg_data = wtg_data[[wtg_pn,time_pn,w_pn,yaw_angle_pn,\
                            P_pn,genspd_pn,\
                            ]+blade_pn_list]
        wtg_data = wtg_data.replace({'\\N':np.NaN})
        for column in [w_pn,yaw_angle_pn,\
                        P_pn,genspd_pn,\
                        ]+blade_pn_list:
            wtg_data[column] = wtg_data[column].astype(float)
        self.wtg_data = wtg_data

        del wtg_data
        self.theory_pw_cur =theory_data
        self.wtg_id = wtg_id
        self.wtg_pn = wtg_pn
        self.time_pn = time_pn
        self.w_pn = w_pn
        self.yaw_angle_pn = yaw_angle_pn
        self.P_pn = P_pn
        # self.temp_pn = temp_pn
        # self.wd_pn = wd_pn
        self.genspd_pn = genspd_pn
        self.blade1_pn = blade1_pn
        self.blade2_pn = blade2_pn
        self.blade3_pn = blade3_pn
        self.blade_pn_list = blade_pn_list
        self.rated_power = rated_power
        self.blade_pn = 'blade_average'
        self.Cp_pn = 'Cp'
        self.month_pn = 'month'

        self.wtg_data[self.time_pn] = pd.to_datetime(self.wtg_data[self.time_pn])
        self.wtg_data = self.wtg_data.sort_values(by=[self.wtg_pn,self.time_pn])
        # self._get_month_list(self.least_amount)
        # if groupby:
        #     self._groupby_minutes()
        # else:
        #     self.wtg_data_1min = self.wtg_data
    

    def _get_month_list(self,least_data=100):
        self.wtg_data[self.month_pn] = self.wtg_data[self.time_pn].dt.month + self.wtg_data[self.time_pn].dt.year*100

        self.month_list = self.wtg_data[[self.month_pn,self.wtg_pn]].groupby(self.month_pn).count().reset_index()
        # 删除少于 10000 的月份的数据
        for m in self.month_list[self.month_list[self.wtg_pn]<least_data][self.month_pn]:
            self.wtg_data = self.wtg_data[self.wtg_data[self.month_pn]!=m].reset_index(drop=True)
        # 重新计算月份的list
        self.month_list = self.wtg_data[[self.month_pn,self.wtg_pn]].groupby(self.month_pn).count().reset_index()

    def _check_month_plot(self,data,flag_pn=None):
        nrows = math.ceil(len(self.month_list)/3)
        fig,axes = plt.subplots(nrows,3,figsize = (30,20))
        if flag_pn:
            flag_list = np.unique(data[flag_pn])
            for i,m_info in self.month_list.iterrows():
                m = m_info[self.month_pn]
                m_data = data[data['month']==m].reset_index(drop=True)
                for j in flag_list:
                    flag_data = m_data[m_data[flag_pn]==j].reset_index(drop=True)
                # print(f'{m}月数据大小为{m_data.shape}')
                    axes[i//3][i%3].scatter(x=flag_data[self.w_pn],y=flag_data[self.P_pn],s=5,label=f'{j} amount {flag_data.shape[0]}')
                axes[i//3][i%3].legend()
                axes[i//3][i%3].set_xlim(0,20)
                axes[i//3][i%3].set_ylim(-10,self.rated_power+100)
                axes[i//3][i%3].set_title(f'{m}月数据{m_data.shape}',fontsize=20)
        else:
            for i,m_info in self.month_list.iterrows():
                m = m_info[self.month_pn]
                m_data = data[data['month']==m].reset_index(drop=True)
                # print(f'{m}月数据大小为{m_data.shape}')
                axes[i//3][i%3].scatter(x=m_data[self.w_pn],y=m_data[self.P_pn],s=5)
                axes[i//3][i%3].set_xlim(0,20)
                axes[i//3][i%3].set_ylim(-10,self.rated_power+100)
                axes[i//3][i%3].set_title(f'{m}月数据{m_data.shape}',fontsize=20)
        plt.close()
        return fig 
    
    def _groupby_minutes(self,) -> list:
        self.wtg_data_1min = self.wtg_data.set_index(self.time_pn).groupby(self.wtg_pn).resample(rule='1T').mean().reset_index()
        self.wtg_data_1min = self.wtg_data_1min.loc[~self.wtg_data_1min['month'].isnull(),:].reset_index(drop=True)
        # print('聚合后数据',self.wtg_data_1min.shape)

    def show_na(self,data):
        fig = plt.figure(figsize=(18,8))
        ax1 = fig.add_subplot(1,2,1)
        missingno.matrix(data, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0),ax=ax1)

        ax2 = fig.add_subplot(1,2,2)
        missingno.bar(data, color="tomato", fontsize=12, ax=ax2)
        return fig
    
    def _fill_na(self,backfill=True):
        # 桨叶角度处理以及填充
        self.wtg_data_1min = self.wtg_data_1min.sort_values(by=[self.wtg_pn,self.time_pn]).reset_index(drop=True)
        self.wtg_data_1min[self.blade_pn] = self.wtg_data_1min[self.blade_pn_list].mean(axis=1)
        if backfill:
            self.wtg_data_1min[self.blade_pn] = self.wtg_data_1min[[self.wtg_pn,self.blade_pn]].groupby([self.wtg_pn]).fillna(method = 'backfill')
        # 功率缺失数据标记
        self.wtg_data_1min = self.wtg_data_1min.loc[~self.wtg_data_1min[self.P_pn].isnull(),:].reset_index(drop=True)
        return

    def draw_blade_wind(self,data,
                        blade_uperbond = 1,
                        bs = 100,
                        blade_scatter_uperbond = 10):
        #画风速-功率散点图，桨叶角度散点图
        fig,axes = plt.subplots(1,3,figsize=(29,8))
        axes[0].scatter(x=data[data[self.w_pn]<10][self.w_pn],
                        y=data[data[self.w_pn]<10][self.P_pn],s=5)
        axes[0].xaxis.set_major_locator(MultipleLocator(0.5))
        axes[1].hist(x=data[data[self.blade_pn]<blade_uperbond][self.blade_pn],bins=bs)
        axes[1].xaxis.set_major_locator(MultipleLocator(0.2))
        axes[2].scatter(x=data[data[self.blade_pn]<blade_scatter_uperbond][self.blade_pn],
                        y=data[data[self.blade_pn]<blade_scatter_uperbond][self.P_pn],
                        s=5)
        return fig


    def working_condition(self,power_blade_cha,
                          blade_thre=1,
                          wind_thre = [4,7]):
        self.wind_lowerbond = wind_thre[0]
        self.wind_uperbond = wind_thre[1]
        # 工况识别
        self.wtg_data_1min['working_condition'] = 'normal power'
        # 不发电工况
        self.wtg_data_1min['working_condition'] = np.where(self.wtg_data_1min[self.P_pn]<1,'1 gen_power<0',self.wtg_data_1min['working_condition'])
        # 功率大于满发功率的工况
        self.wtg_data_1min['working_condition'] = np.where((self.wtg_data_1min[self.P_pn]>=self.rated_power)&\
                                                           (self.wtg_data_1min['working_condition'] == 'normal power'),\
                                                            f'2 gen_power>={self.rated_power}',\
                                                            self.wtg_data_1min['working_condition'])
        # 桨叶角度大于30的工况
        self.wtg_data_1min['working_condition'] = np.where((self.wtg_data_1min[self.blade_pn]>30)&\
                                                           (self.wtg_data_1min['working_condition'] == 'normal power'),\
                                                            '3 blade_angle>30',\
                                                            self.wtg_data_1min['working_condition'])
        # 限电工况
        self.wtg_data_1min['working_condition'] = np.where((self.wtg_data_1min[self.blade_pn]>blade_thre)&\
                                                           (self.wtg_data_1min['working_condition'] == 'normal power')&\
                                                            (self.wtg_data_1min[self.P_pn]<power_blade_cha),\
                                                            f'4 limit_power',\
                                                            self.wtg_data_1min['working_condition']) 
        # 风速非爬坡的工况
        self.wtg_data_1min['working_condition'] = np.where(((self.wtg_data_1min[self.w_pn]<wind_thre[0])|\
                                                           (self.wtg_data_1min[self.w_pn]>wind_thre[1]))&\
                                                           (self.wtg_data_1min['working_condition'] == 'normal power'),\
                                                            '5 beyond climbing stage',\
                                                            self.wtg_data_1min['working_condition'])      

        self.wtg_normal_power= self.wtg_data_1min[self.wtg_data_1min['working_condition']=='normal power'].reset_index(drop=True)
        




    def clean_data(self,
                   difference=True,
                   difference_thre = 1000,
                   sigma_3=True,
                   sigma_times=2,
                   Cp=True,
                   Cp_thre = [-10e3,10e5],
                   ):
        self.wtg_normal_power['nan_marker'] = 'useful data'
        if difference:
            self.clean_based_on_difference(difference_thre)
        if sigma_3:
            self.clean_based_on_Power_wind_3sigma(sigma_times)
        if Cp:
            self.clean_based_on_Cp(Cp_thre)
        self.wtg_use = self.wtg_normal_power[self.wtg_normal_power['nan_marker']=='useful data'].reset_index(drop=True)

    def clean_based_on_difference(self,threshold=1000):
        self.wtg_normal_power['pw_diff'] = self.wtg_normal_power[[self.wtg_pn,self.blade_pn]].groupby([self.wtg_pn]).diff()
        self.wtg_normal_power['nan_marker'] = np.where(self.wtg_normal_power['pw_diff']>threshold,'1 abnormal_jump',self.wtg_normal_power['nan_marker'])
        return
    
    def check_Cp(self):
        self.wtg_normal_power[self.Cp_pn] = self.wtg_normal_power[self.P_pn]/np.where(self.wtg_normal_power[self.w_pn]<1,1,self.wtg_normal_power[self.w_pn]**3)
        fig,ax = plt.subplots(figsize = (16,8))
        # ax.scatter(x=wtg_data_1min[blade1_pn],y=wtg_data_1min[P_pn])
        _ = ax.hist(x=self.wtg_normal_power[self.Cp_pn],bins=100)
        ax.set_title('Cp的分布情况',size=20)
        return fig
    
    def clean_based_on_Cp(self,Cp_threshold):
        # self.wtg_normal_power[self.Cp_pn] = self.wtg_normal_power[self.P_pn]/np.where(self.wtg_normal_power[self.w_pn]<1,1,self.wtg_normal_power[self.w_pn]**3)
        self.wtg_normal_power['nan_marker'] = np.where(((self.wtg_normal_power[self.Cp_pn]<=Cp_threshold[0])|\
                                                      (self.wtg_normal_power[self.Cp_pn]>Cp_threshold[1]))&\
                                                        (self.wtg_normal_power['nan_marker'] == 'useful data'),
                                                   '2 abnormal_Cp',
                                                   self.wtg_normal_power['nan_marker'])
        
        return

    def clean_based_on_Power_wind_3sigma(self,sigma_times):
        def OutlierDetection(df, pn):
            # 计算均值
            u = df[pn].mean()
            # 计算标准差
            std = df[pn].std()
            # 定义3σ法则识别异常值
            outliers = df[np.abs(df[pn] - u) > 3* std].index
            # 剔除异常值，保留正常的数据
            # outliers_portion = round(len(outliers)/df.shape[0],4)
            # print(f'异常值有{len(outliers)},占比{outliers_portion*100}%')
            # 返回异常值的index
            return outliers
        self.bs_sigma = np.arange(self.wind_lowerbond,self.wind_uperbond+0.1,0.1)
        self.ls_sigma = []
        for i,b in enumerate(self.bs_sigma):
            if i>0:
                self.ls_sigma.append(round((b+self.bs_sigma[i-1])/2,2))
        # ls = np.linspace(4,8.2,45)
        self.wtg_normal_power['wind_bin'] = pd.cut(self.wtg_normal_power[self.w_pn],bins=self.bs_sigma,labels=self.ls_sigma,right=False,).astype(float)
        # self.ls = np.unique(self.wtg_normal_power['wind_bin'])
        for i in range(sigma_times):
            index_list = []
            for _,m_info in self.month_list.iterrows():
                m = m_info[self.month_pn]
                month_data = self.wtg_normal_power[(self.wtg_normal_power[self.month_pn]==m)&(self.wtg_normal_power['nan_marker']=='useful data')]
                # print('#'*10,f'月份{m},数据大小{month_data.shape[0]}')
                for w in self.ls_sigma:
                    w_data = month_data[month_data['wind_bin'] == w]
                    # print(f'风速仓{w}m/s,数据大小{w_data.shape[0]}')
                    if w_data.shape[0]<=0:
                        continue
                    o_index = OutlierDetection(w_data,self.P_pn)
                    index_list += list(o_index)
            self.wtg_normal_power.loc[index_list,'nan_marker'] = f'3 sigma outlier round {i+1}'
        return


    def _divide_float_to_bin(self,bin_len=0.5):
        self.bs_use = np.arange(self.wind_lowerbond,self.wind_uperbond+bin_len,bin_len)
        self.ls_use = []
        for i,b in enumerate(self.bs_use):
            if i>0:
                self.ls_use.append(round((b+self.bs_use[i-1])/2,2))
        self.wtg_use['wind_bin'] = pd.cut(self.wtg_use[self.w_pn],bins=self.bs_use ,right=False,labels=self.ls_use).astype(float)
        return

    def drop_yaw_outlier(self,yaw_angle_lo = -30,
                         yaw_angle_hi=30):
        self.wtg_use = self.wtg_use[self.wtg_use[self.yaw_angle_pn] <= yaw_angle_hi].reset_index(drop=True)
        self.wtg_use = self.wtg_use[self.wtg_use[self.yaw_angle_pn] >= yaw_angle_lo].reset_index(drop=True)
        return
        
    

    def quantreg(self,least_samples = 2000,
                 q=0.5,
                 covariant='robust',
                 kernel = 'epa',
                 bwidth ='hsheather',
                 max_iter = 5000,
                 tolerance = 1e-6
                 ):
        h_list = []
        m_list = []
        fig_list = []
        for i,m_info in self.month_list.iterrows():
            if i==0:
                continue
            m2 = m_info[self.month_pn]
            m1 = self.month_list.loc[i-1,self.month_pn]
            m_data =  self.wtg_use[(self.wtg_use['month']==m1)|(self.wtg_use['month']==m2)].reset_index(drop=True)
            print(f'{m1},{m2}月数据大小为{m_data.shape}')
            if m_data.shape[0]==0:
                continue
            nrows = math.ceil(len(self.ls_use)/3)
            # print(nrows)
            fig,axes = plt.subplots(nrows,3,figsize = (20,nrows*5+2))
            mh_list = []
            m_list.append(f'{m1},{m2}月')
            for i,v in enumerate(self.ls_use):
                v_data = m_data[m_data['wind_bin']==v].reset_index(drop=True)
                # print(f'数据大小为{v_data.shape}')
                v_data['x1'] = v_data[self.yaw_angle_pn]
                v_data['x2'] = v_data[self.yaw_angle_pn]**2
                v_data['y'] = v_data[self.P_pn]
                X = v_data[['x1','x2']]
                y = v_data['y']
                # constraints = 'b2 < 0'
                reg = smf.quantreg("y ~ x1 + x2",v_data)
                res = reg.fit(q=q,
                              vcov = covariant,
                              kernel = kernel,
                              bandwidth=bwidth,
                              max_iter=max_iter,
                              p_tol = tolerance)
                # print(res.summary())
                preds = res.predict(X)
                p1 = res.params[1]
                p2 = res.params[2]
                h = -p1/(2*p2)
                axes[i//3][i%3].scatter(x=v_data[self.yaw_angle_pn],y=v_data[self.P_pn],s=5)
                
                axes[i//3][i%3].set_title(f'{v}m/s 极值点{round(h,2)} \n数据量{v_data.shape[0]},a={round(p2,4)}',\
                            fontsize=20)
                # axes[i//3][i%3].set_xlim(-30,30)
                axes[i//3][i%3].set_ylim(-10,self.rated_power+100)
                if (abs(h)<30) and (v_data.shape[0]>least_samples) and (p2<0):
                    mh_list.append(h)
                    axes[i//3][i%3].scatter(v_data[self.yaw_angle_pn],preds,color='red',s=0.5)
                else:
                    # print(abs(h),v_data.shape[0])
                    # print((abs(h)<30),v_data.shape[0]>2000)
                    mh_list.append(np.NaN)
                # print(mh_list)
            h_list.append(mh_list)
            fig.suptitle(f'{m1},{m2}月',fontsize=20)
            plt.tight_layout()
            fig_list.append(fig)
            
            # plt.savefig(ROOT_PATH + f'{m}月.jpg',bbox_inches='tight',facecolor='white',dpi=500)

        result = pd.DataFrame(h_list).T
        result.columns = m_list
        result[self.w_pn] = self.ls_use

        results_ls = []
        for m in m_list:
            result_m = result[[m,self.w_pn]].dropna().reset_index(drop=True)
            angle = (result_m[m]*(result_m[self.w_pn]**3)).sum()/(result_m[self.w_pn]**3).sum()
            results_ls.append(angle)
        result.loc[len(result.index)] = results_ls+[0]
        return fig_list,result

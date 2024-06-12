import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io 
import zipfile


def save_figures(path,figure,filename):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path,filename)
    figure.savefig(save_path,bbox_inches='tight',dpi=500,facecolor='white')
def save_data(path,data,filename):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path,filename)
    data.to_excel(save_path,index=False)

# create zip file from list of figs
def figs2zip(figs,fig_title) -> bytes:
    """THIS WILL BE RUN ON EVERY SCRIPT RUN"""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(file=zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        for i,fig in enumerate(figs):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            filename = f'{fig_title}{i}.png'
            z.writestr(zinfo_or_arcname=filename, data=buf.getvalue() )
            buf.close()
    
    buf = zip_buf.getvalue()
    zip_buf.close()
    return buf
import os, glob
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
from matplotlib import colors
from tqdm import tqdm 


def psd2im(df,
           ax=None,
           fig=None,
           mask=None,
           savefp=None,
           dpi=600,
           n_xticks=5,
           figsize=(1.6, 1.2),
           show_figure=True,
           use_title=False,
           title=None,
           fit_data=None,
           index=None,
           vmax=1e4,
           lcolor='white',
           lwidth=3,
           use_xaxis=True,
           use_xlabel=False,
           use_yaxis=True,
           use_cbar=False,
           ftsize=16
          ):
    """Draw single or multiple surface plots.

    Parameters:
        df (dataframe)     --  particle size distribution data for one day or multiple days
        ax (ax)            --  specify the ax to visualize the psd. If not specified, a new one will be created
        fig (fig)          --  the whole figure
        mask (array)       --  numpy array with the same shape as the input psds
        savefp (str)       --  path for storing the figures
        dpi (int)          --  default is 600
        n_xticks (int)     --  how many ticklabels shown on the x-axis
        figsize (tuple)    --  used only if a new ax is created
        show_figure (bool) --  clear all the figures if drawing many surface plots
        use_title (bool)   --  use the date as the title for the psd
        title (str)        --  the specified title
        fit_data (list)    --  the fitted time points and related Dps
        index (int)        --  there many be more than one masks detected for one day's psd
        vmax (int)         --  color scale for visualization, default is 1e4.
        lcolor (str)       --  color for visualizing the GR
        lwidth (int)       --  linewidth
        use_xaxis (bool)   --  whether to draw the x-axis
        use_yaxis (bool)   --  whether to draw the y-axis
        use_cbar (bool)    --  whether to use the colorbar
        ftsize (int)       --  fontsize for plotting
    
    """

    # get the psd data
    dfc = df.copy(deep=True)    # get a copy version
    df_min = np.nanmin(dfc.replace(0, np.nan))    # find the minimul value
    dfc.fillna(df_min, inplace=True)    # use the minimul value to replace the na values
    dfc[dfc == 0] = df_min               # use the minimul value to replace 0
    dfc = dfc.replace(0, df_min)

    values = dfc.values.T if mask is None else (dfc.values*mask).T   # values for visualization
    dps = [float(dp)*1e-9 for dp in list(dfc.columns)]    # Dps
    tm = dfc.index.values    # time points

    # check how many days of data to be shown
    whole_dates = np.unique([item.date() for item in df.index])
    num_days = (whole_dates[-1] - whole_dates[0]).days + 1

    # once the ax is none, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(tm,    # time points
                      dps,    # particle sizes
                      values, # distribution
                      norm=colors.LogNorm(vmin=1e1, vmax=vmax),
                      cmap='jet',
                      shading='auto')

    # add the fitted line for determinating the GRs
    if fit_data is not None:
        ax.plot(fit_data[0], fit_data[1], c=lcolor, linewidth=lwidth)

    # use the log scale for y-axis
    ax.set_yscale('log')

    # get title
    if title is None:
        title = str(whole_dates[0]) if num_days <= 1 else str(whole_dates[0])+'_'+str(whole_dates[-1])

    # add index
    if index is not None:
        title = title + f' {index}'

    # add the title on the figure
    if use_title:
        ax.set_title(title, fontsize=ftsize+4)

    # add y-axis
    if use_yaxis:
        ax.set_ylabel('$\mathrm{D_p}$ (m)', fontsize=ftsize+2)
    else:
        ax.get_yaxis().set_visible(False)

    # add x-axis
    if use_xaxis:
        xtick = [datetime(sdate.year, sdate.month, sdate.day) + timedelta(i/(n_xticks-1)) for sdate in whole_dates
                 for i in range(n_xticks-1)] + [datetime(whole_dates[-1].year, whole_dates[-1].month, whole_dates[-1].day)+timedelta(1)]
        xtick_labels = ['00:00', '06:00', '12:00',  '18:00'] * num_days + ['00:00']
        ax.set_xticks(xtick)
        ax.set_xticklabels(xtick_labels)
    else:
        ax.get_xaxis().set_visible(False)

    if use_xlabel:
        ax.set_xlabel('Local time (h)', fontsize=ftsize+2)

    # add colorbar
    if use_cbar:
        cbar = fig.colorbar(im, ax=ax)    # here fig is the default input for subplots
        cbar.set_label('dN/dlog$\mathrm{D_p} (\mathrm{cm}^{-3})$', fontsize=ftsize+2)

    # to avoid the black edges
    if (not use_xaxis) and (not use_yaxis):
        ax.set_axis_off()

    # save the currect figure
    if (savefp is not None) and (not use_xaxis) and (not use_yaxis):
        fig.savefig(os.path.join(savefp, title +'.png'), bbox_inches='tight', pad_inches=0, dpi=dpi)
    elif (savefp is not None) and (use_xaxis or use_yaxis):
        fig.savefig(os.path.join(savefp, title +'.png'), bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    if not show_figure:
        plt.cla()
        plt.clf()
        plt.close('all')
    return im


def round_ten(t, rounding=True):
    """
    https://stackoverflow.com/a/34054489
    """
    if rounding:
        t = t + pd.Timedelta('5 minutes')  # rounding
    return t.replace(minute=t.minute//10*10).replace(second=0)


def get_nearby_day(current_day, direction=1):
    r"""
    Get the next day given the current day.
    """
    return str(pd.to_datetime(current_day) + timedelta(days=direction))[:10]


def savedf(df, savefp):
    """
    save dataframe to savefp
    """
    dfc = df.copy(deep=True)
    df_min = np.nanmin(dfc.replace(0, np.nan))
    dfc.fillna(df_min, inplace=True)
    dfc[dfc == 0] = df_min
    dfc = dfc.replace(0, df_min)
    dfc.to_csv(savefp)
    # title = savefp.split('.')[0]
    # psd2im(dfc, figsize=(12, 5), show_figure=False, savefp='.', use_xaxis=True, use_yaxis=True, use_cbar=True, use_title=True, title=title)
    


if __name__ == '__main__':
    import argparse
    # for X in loader:
    #     print(X)
    parser = argparse.ArgumentParser('gNPF')

    # datasets
    parser.add_argument('--dataroot', type=str, default='datasets', help='folder that stores the datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=64, help='mini batch size')
    parser.add_argument('--station', type=str, default='hyy', help='station name [var | hyy | kum]')
    # parser.add_argument('--vmax', type=float, default=1e6, help='maximum value for normalization')
    args = parser.parse_args()

    savefp = os.path.join(args.dataroot, args.station)
    os.makedirs(savefp, exist_ok=True)

    # file = 'kumpula.csv'
    # name = 'kum'
    # folder = 'files'
    # os.makedirs(folder, exist_ok=True)
    df = pd.read_csv(savefp+'.csv', parse_dates=[0], index_col=0)
    df = df.iloc[:, 9:]
    days = sorted(np.unique(df.index.date.astype(str)).tolist())
    # print(len(days))

    # clean the timepoints
    df.index = df.index.map(lambda x: round_ten(x, rounding=False))

    # save the first day
    df_1 = df[df.index.date.astype(str) == days[0]]
    if df_1.shape[0] == 144:
        savedf(df_1, f'{savefp}/0.csv')

    # save the last day
    df_end = df[df.index.date.astype(str) == days[-1]]
    if df_end.shape[0] == 144:
        savedf(df_end, f'{savefp}/end.csv')

    total_index = df.index
    # get continuous csv files
    for idx, day in enumerate(tqdm(days)):
        df_ = df.loc[day]
        # print(df_)
        dim = df_.shape[0]
        if dim == 144:
            prev_day = get_nearby_day(day, -1) + ' 23:00:00'
            next_day = get_nearby_day(day, 1) + ' 00:50:00'
            ind = (total_index >= prev_day ) & (total_index <= next_day )
            # print(ind)
            if sum(ind) == 156:
                for i in range(13):
                    df_ = df[ind].iloc[i:i+144, :]
                    savedf(df_, f'{savefp}/{idx}_{i}.csv')
            else:
                savedf(df_, f'{savefp}/{idx}.csv')


    # for file in glob.glob('files/*.csv')[:3000]:
    #     df = pd.read_csv(file, parse_dates=[0], index_col=0)
    #     title = file.split('.')[0]
    #     psd2im(df, savefp='figures', figsize=(12, 5), show_figure=False, use_xaxis=True, use_yaxis=True, use_cbar=True, use_title=True, title=title)

    # max_val = 0

    # for file in glob.glob('files/*.csv'):
    #     values = pd.read_csv(file, parse_dates=[0], index_col=0).values
    #     if np.max(values) > max_val:
    #         max_val = np.max(values)
    #         print(max_val)
            
            



    # df_new = []    
    # for day in days:
    #     df_ = df[df.index.date.astype(str) == day]
    #     if df_.shape[0] == 144:
    #         df_.index = pd.date_range(day + ' 00:00', day + ' 23:50', freq='10min')
    #         df_new.append(df_)
    #     else:
    #         df_.index = df_.index.map(lambda x: round_ten(x, rounding=False))
    # print(len(df_new))
    # df = pd.concat(df_new)
    # print(df.head())

    
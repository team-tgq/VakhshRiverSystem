import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# ==================== 界面组件 ====================
class WarningBarChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8,4), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.warning_levels = ['蓝','黄','橙','红']
        self.colors = ['#1f77b4','yellow','#ff9900','#d62728']

    def update_chart(self, station_df):
        self.ax.clear()
        level_counts = station_df['预警等级'].value_counts().reindex(self.warning_levels, fill_value=0)
        bars = self.ax.bar(level_counts.index, level_counts.values, color=self.colors, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x()+bar.get_width()/2., height+0.1, f'{int(height)}', ha='center', va='bottom')
        self.ax.set_title('预警等级分布', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('预警等级')
        self.ax.set_ylabel('站点数量')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.fig.tight_layout()
        self.draw()

class TimeSeriesChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10,5), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

    def update_chart(self, weather_data, hydro_data):
        self.ax1.clear()
        self.ax2.clear()
        if weather_data.empty or hydro_data.empty:
            return
        weather_data['时间'] = pd.to_datetime(weather_data['时间'])
        hydro_data['时间'] = pd.to_datetime(hydro_data['时间'])
        line1 = self.ax1.plot(weather_data['时间'], weather_data['降雨量(mm/h)'],
                              label='降雨量', color='blue', linewidth=2, marker='o', markersize=4)
        line2 = self.ax2.plot(hydro_data['时间'], hydro_data['水位(m)'],
                              label='水位', color='green', linewidth=2, marker='s', markersize=4)
        self.ax1.set_xlabel('时间')
        self.ax1.set_ylabel('降雨量 (mm/h)', color='blue')
        self.ax2.set_ylabel('水位 (m)', color='green')
        self.ax1.set_title('降雨量与水位变化趋势', fontsize=14, fontweight='bold')
        self.ax1.grid(True, alpha=0.3, linestyle='--')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        self.ax1.legend(lines, labels, loc='upper left')
        self.fig.autofmt_xdate(rotation=45)
        self.fig.tight_layout()
        self.draw()

class SpatialMapChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8,6), dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.warning_colors = {'蓝':'#1f77b4','黄':'yellow','橙':'#ff9900','红':'#d62728'}

    def update_chart(self, station_df):
        self.ax.clear()
        if station_df.empty:
            return
        for level, color in self.warning_colors.items():
            level_data = station_df[station_df['预警等级'] == level]
            if not level_data.empty:
                scatter = self.ax.scatter(level_data['经度'], level_data['纬度'],
                                          s=level_data['等级值']*40, c=color,
                                          label=level, alpha=0.8, edgecolors='black')
                for _, row in level_data.iterrows():
                    self.ax.annotate(row['站点名称'], xy=(row['经度'], row['纬度']),
                                     xytext=(5,5), textcoords='offset points', fontsize=8)
        self.ax.set_title('站点空间分布', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('经度')
        self.ax.set_ylabel('纬度')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.legend(title='预警等级')
        self.ax.set_xlim(110,125)
        self.ax.set_ylim(35,45)
        china_box = Rectangle((110,35), 15, 10, linewidth=2, edgecolor='gray', facecolor='none', alpha=0.5)
        self.ax.add_patch(china_box)
        self.fig.tight_layout()
        self.draw()